from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.hptrack import build_hptrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
import os


class HPTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(HPTRACK, self).__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = build_hptrack(params.cfg,training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu',
                                           weights_only=False)['net'],strict=True)
        self.cfg = params.cfg
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        if self.cfg.TEST.WINDOW == True:
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.debug = params.debug
        self.frame_id = 0
        self.his_chp_tokens=[None]*self.cfg.MODEL.BACKBONE.NUM_MFHP_LAYERS
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPT, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPT[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPT.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        if hasattr(self.cfg.TEST.INTER, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.INTER[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.INTER.DEFAULT
        print("Update intervals is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.MB, DATASET_NAME):
            self.memory_bank = self.cfg.TEST.MB[DATASET_NAME]
        else:
            self.memory_bank = self.cfg.TEST.MB.DEFAULT
        print("template_pool is: ", self.memory_bank)

    def initialize(self, image, info: dict):
        if self.debug == 2:
            self.save_path = os.path.join(self.save_dir, info['seq_name'])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(
            image, info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size)

        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.state = info['init_bbox']

        prev_box_crop = transform_image_to_crop(
            torch.tensor(info['init_bbox']),torch.tensor(info['init_bbox']),
            resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True)

        self.template = [template]*self.num_template
        self.anno=[prev_box_crop.to(template.device).unsqueeze(0)]*self.num_template
        self.memorys=[]
        self.frame_id = 0
    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(
            image, self.state,
            self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]
        template_list = self.template.copy()
        anno_list=self.anno.copy()
        with torch.no_grad():
            his_chp_tokens=self.his_chp_tokens
            back_out,his_chp_tokens = self.network.forward_backbone(template_list,search_list,anno_list,his_chp_tokens)
        # run the head
        with torch.no_grad():
            out_dict = self.network.forward_head(back_out=back_out,)
        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map

        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.head.head_bbox.cal_bbox(
                response,
                out_dict['size_map'],
                out_dict['offset_map'],
                return_score=True)
        else:
            pred_boxes, conf_score = self.network.head.head_bbox.cal_bbox(
                response,out_dict['offset_map'],
                return_score=True)

        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf=conf_score.item()
        self.his_chp_tokens = his_chp_tokens
        if self.num_template > 1 :
            # template injection process
            if  conf > self.update_threshold:
                z_patch_arr, resize_factor = sample_target(image, self.state,self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),torch.tensor(self.state),resize_factor,
                                    torch.Tensor([self.params.template_size, self.params.template_size]),normalize=True)
                self.memorys.append ((conf,template.to(self.device),prev_box_crop.to(self.device).unsqueeze(0)))

                while len(self.memorys) > self.memory_bank:
                    min_score, min_template, min_anno = min(self.memorys, key=lambda x: x[0])
                    self.memorys.remove((min_score, min_template,min_anno))
                    self.memorys.pop(0)
            #template extraction process
            if self.frame_id % self.update_intervals == 0 and  len(self.memorys)>=self.num_template-1:
                self.select_memory_frames()
                #self.select_memory_frames2()

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)
        elif self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_path, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        return {"target_bbox": self.state,
                "best_score": conf}
    def select_memory_frames(self):
        new_template_len = self.num_template-1
        total = len(self.memorys)
        assert total >= new_template_len
        split_idx = np.linspace(0, total, new_template_len + 1).astype(int)
        for i in range(new_template_len):
            start = split_idx[i]
            end = split_idx[i + 1]
            group = self.memorys[start:end]
            best_score, best_template,best_anno = max(group, key=lambda x: x[0])
            self.template.append(best_template)
            self.anno.append(best_anno)
            self.template.pop(1)
            self.anno.pop(1)
        assert len(self.template) == self.num_template == len(self.anno)
    def select_memory_frames2(self):
        new_template_len = self.num_template - 1
        total = len(self.memorys)
        assert total >= new_template_len
        split_idx = np.linspace(0, total, new_template_len + 1).astype(int)
        for i in range(new_template_len):
            start = split_idx[i]
            end = split_idx[i + 1]
            center_idx = (start + end) // 2
            center_score, center_template, center_anno = self.memorys[center_idx]
            self.template.append(center_template)
            self.anno.append(center_anno)
            self.template.pop(1)
            self.anno.pop(1)
        assert len(self.template) == self.num_template == len(self.anno)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return HPTRACK
