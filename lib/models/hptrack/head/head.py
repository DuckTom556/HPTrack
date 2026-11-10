from torch import nn
from lib.models.hptrack.head.head_bbox import MLPPredictor,Corner_Predictor,CenterPredictor
from lib.models.hptrack.head.head_conf import ConfidenceHead,ScoreDecoder
#from lib.models.hptrack.head.head_conf import ScoreDecoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
import math
class HEAD(nn.Module):

    def __init__(self,head_bbox,head_conf,head_type):
        super(HEAD, self).__init__()
        self.head_bbox = head_bbox
        self.head_conf=head_conf
        self.head_type=head_type
    def forward(self, back_out,lens_cls,lens_z,lens_x,gt_score_map=None):
        x=back_out[:,-lens_x:,:]
        fx_sz = int(math.sqrt(lens_x))
        fz_sz = int(math.sqrt(lens_z))
        output,pre_bbox=self.forward_head_bbox(back_out_search=x,fx_sz=fx_sz,fz_sz=fz_sz,gt_score_map=gt_score_map)
        if self.head_conf is not None:
            cls = back_out[:, :lens_cls, :]
            B,_,C=back_out.shape
            H_s = W_s = int(math.sqrt(x.size(1)))
            bbox = pre_bbox.clone().detach()
            x_2d = x.transpose(1, 2).reshape(B, C, H_s, W_s)
            conf = self.head_conf(cls,x_2d,bbox)
        else:
            conf=None
        output['confidence'] = conf
        return output

    def forward_head_bbox(self, back_out_search,fx_sz,fz_sz, gt_score_map=None):
        feature = back_out_search
        bs, HW, C = feature.size()
        if self.head_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, fx_sz, fx_sz)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.head_bbox(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,}
            return out,outputs_coord_new
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head_bbox(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,}
            return out,outputs_coord_new
        elif self.head_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.head_bbox(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map,}
            return out,outputs_coord_new
        else:
            raise NotImplementedError


def build_head(cfg,dim):
    num_channels_enc = dim
    stride = cfg.MODEL.BACKBONE.STRIDE
    if cfg.MODEL.HEAD.TYPE == "MLP":
        in_channel = num_channels_enc
        hidden_dim = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        head_bbox = MLPPredictor(inplanes=in_channel, channel=hidden_dim,
                                feat_sz=feat_sz, stride=stride)
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            head_bbox = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = num_channels_enc
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        head_bbox = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
    if cfg.MODEL.BACKBONE.USE_POOL:
        head_conf = ScoreDecoder(pool_size=4, hidden_dim=dim, num_heads=dim//64)
    elif cfg.MODEL.BACKBONE.USE_CONF:
        head_conf=ConfidenceHead(dim)
    else:
        head_conf=None
    return HEAD(head_bbox, head_conf,head_type=cfg.MODEL.HEAD.TYPE)