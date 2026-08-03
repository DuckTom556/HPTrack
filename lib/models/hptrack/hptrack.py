"""
HPTrack Model
"""
import math
from torch import nn
from .backbone.hivit import build_backbone
from .head.head import build_head

class HPTrack(nn.Module):
    """ This is the base class for HPTrack """
    def __init__(self, backbone, head,cfg,):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            head: torch module of the head architecture. See head.py
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_patch_x=backbone.num_patches_search
        self.num_patch_z=backbone.num_patches_template
        self.num_main_blocks=backbone.num_main_blocks
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))
        self.freeze_en = cfg.MODEL.BACKBONE.FREEZE
        self.num_frames = cfg.DATA.SEARCH.NUMBER
        self.num_template = cfg.DATA.TEMPLATE.NUMBER
        self.his_len=None
    def forward(self, template_list=None, search_list=None,anno_list=None,back_out=None,
                his_search=None,his_hp=None,mode="backbone"):
        if mode == "backbone":
            return self.forward_backbone(template_list, search_list,anno_list,his_search,his_hp)
        elif mode == "head":
            return self.forward_head(back_out)
        else:
            raise ValueError
    def forward_backbone(self, template_list, search_list,anno_list,his_search,his_hp):
        # Forward the backbone
        xz,his_search,his_hp= self.backbone(template_list, search_list,anno_list,his_search,his_hp)
        xz = self.backbone.norm_(xz)
        return xz,his_search,his_hp
    def forward_head(self, back_out, gt_score_map=None):
        lens_z=self.num_patch_z
        lens_x=self.num_patch_x
        lens_cls=1
        output=self.head(back_out,lens_cls,lens_z,lens_x,gt_score_map)
        return output
def build_hptrack(cfg,training=True):
    backbone=build_backbone(cfg,training)
    head = build_head(cfg, backbone.embed_dim)
    model = HPTrack(backbone,head,cfg,)
    return model
