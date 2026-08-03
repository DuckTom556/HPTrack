from functools import partial
import torch
import torch.nn as nn
from lib.models.hptrack.backbone.vim import build_vim
from lib.models.hptrack.backbone.ssm import build_ssm

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.embed_dim = None
        self.num_patches_search = None
        self.num_patches_template = None
        self.pos_embed_z = None
        self.pos_embed_x = None
        self.grad_ckpt = False
        self.template_background_token = None
        self.template_foreground_token = None
        self.search_token = None
        self.token_type_indicate = False
        self.mamba_t = None
        self.mamba_s = None
        self.use_mamba_t = False
        self.use_mamba_s = False
        self.layer_indexes = []
        self.use_conf = False
        self.num_layers = None
        self.hp_token = None
        self.use_hp = False
        self.cls_pos_embed=None
        self.hp_pos_embed = None
        self.use_interaction = False
        self.interaction = None
    def finetune_track(self, cfg, patch_start_index=1):
        search_size = cfg.DATA.SEARCH.SIZE
        template_size = cfg.DATA.TEMPLATE.SIZE
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE
        self.layer_indexes = cfg.MODEL.BACKBONE.LAYER_INDEXES
        self.num_layers = cfg.MODEL.BACKBONE.NUM_LAYERS
        self.use_mamba_t = cfg.MODEL.BACKBONE.USE_MAMBA_T
        self.use_mamba_s = cfg.MODEL.BACKBONE.USE_MAMBA_S
        self.use_conf = cfg.MODEL.BACKBONE.USE_CONF
        self.use_hp = cfg.MODEL.BACKBONE.USE_HP
        self.return_inter = cfg.MODEL.BACKBONE.RETURN_INTER
        self.token_type_indicate = cfg.MODEL.BACKBONE.TOKEN_TYPE_INDICATE
        if self.use_mamba_t:
            self.mamba_t = nn.ModuleList([build_vim(embed_dim=self.embed_dim) for _ in range(self.num_layers)])
        if self.use_mamba_s:
            self.mamba_s=build_ssm(d_model=self.embed_dim,d_state=cfg.MODEL.BACKBONE.D_STATE,grad_ckpt=self.grad_ckpt)
        if self.use_hp:
            '''self.hp_token=nn.Parameter(torch.zeros(1, 1, self.embed_dim))'''
            self.hp_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.embed_dim)) for _ in range(self.num_layers)])
        self.num_patches_search = (search_size// new_patch_size) * (search_size// new_patch_size)
        self.num_patches_template = (template_size// new_patch_size) * (template_size// new_patch_size)
        if self.token_type_indicate:
            self.template_background_token = nn.Parameter(torch.zeros(self.embed_dim))
            self.template_foreground_token = nn.Parameter(torch.zeros(self.embed_dim))
            self.search_token = nn.Parameter(torch.zeros(self.embed_dim))
        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size // self.patch_size, self.img_size// self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        # for search region
        new_P_H, new_P_W = search_size // new_patch_size, search_size // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)
        # for template region
        new_P_H, new_P_W = template_size // new_patch_size, template_size // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        # for cls token (keep it but not used)
        if self.use_conf and self.cls_token is not None :
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)
        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def create_mask(self, image, image_anno):
        height = image.size(2)
        width = image.size(3)
        # Extract bounding box coordinates
        x0 = (image_anno[:, 0] * width).unsqueeze(1)
        y0 = (image_anno[:, 1] * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)
        # Generate pixel indices
        x_indices = torch.arange(width, device=image.device)
        y_indices = torch.arange(height, device=image.device)
        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()
        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)

        return mask
    def prepare_masks(self,z,z_anno):
        z_indicate_mask = self.create_mask(z, z_anno)
        z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size,self.patch_size)  # to match the patch embedding
        z_indicate_mask = z_indicate_mask.mean(dim=(3, 4)).flatten(1)
        template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(
            z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
        template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(
            z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
        weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
        weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
        z_indicate = weighted_foreground + weighted_background
        return z_indicate
    def my_layer(self,x, blocks):
        for idx, blk in enumerate(blocks):  # backbone block
            x,att =blk(x)
        return x
    def forward_features(self, template_list, search_list, template_anno_list,his_search,his_hp):
        num_template = len(template_list)
        z = torch.stack(template_list, dim=1)  # (b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])  # (bn,c,h,w)
        x = torch.stack(search_list, dim=1)  # (b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])  # (bn,c,h,w)
        if self.token_type_indicate:
            z_anno = torch.stack(template_anno_list, dim=1)  # (b,n,4)
            z_anno = z_anno.view(-1, *z_anno.size()[2:])  # (bn,4)
            z_indicate=self.prepare_masks(z,z_anno)
        z = self.patch_embed(z)
        x = self.patch_embed(x)
        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)
        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]
        z += self.pos_embed_z
        x += self.pos_embed_x
        if self.token_type_indicate:
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            x = x + x_indicate
            z = z + z_indicate

        z = z.view(-1, num_template, z.size(-2), z.size(-1))  # b,n,l,c
        # HTFA process
        if self.use_mamba_s:
            z=self.mamba_s(z)  # b,l,c
        else:
            z = z.reshape(z.size(0), -1, z.size(-1))  # b,l,c
        x = x.reshape(x.size(0), -1, x.size(-1))
        zx = torch.cat([z, x], dim=1)
        #if self.use_conf and self.cls_token is not None:
        if self.cls_token is not None and self.use_conf:
            B = zx.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens+=self.cls_pos_embed
            #cls_tokens = cls_tokens if his_hp[0] is None else his_hp[0] + cls_tokens
            zx = torch.cat([cls_tokens, zx], dim=1)
        if self.hp_token is not None:
            '''B = zx.shape[0]
            hp_tokens = self.hp_token.expand(B, -1, -1)
            hp_tokens =hp_tokens+ self.hp_pos_embed
            #hp_tokens=hp_tokens if his_hp[1] is  None else hp_tokens+his_hp[1]'''
            #zx = torch.cat([zx, hp_tokens], dim=1)
            hp_tokens=[]
            B = zx.shape[0]
            for i in range(len(self.hp_token)):
                hp_token= self.hp_token[i].expand(B, -1, -1)
                hp_token= hp_token if his_hp[i] is None else his_hp[i] + hp_token
                hp_tokens.append(hp_token)
        zx = self.pos_drop(zx)
        # MFHP process
        for i, index in enumerate(self.layer_indexes):
            if self.use_hp :
                zx=torch.cat([zx,hp_tokens[i]], dim=1)
            zx=self.my_layer(zx,self.blocks[index[0]:index[1]])
            if self.use_mamba_t:
                zx, his_hp[i] = self.mamba_t[i](zx, self.num_patches_search, 1)#his_hp[i]
            elif self.use_hp:
                his_hp[i]=zx[:,-1:,:]
                zx=zx[:,:-1,:]
        return zx,his_search,his_hp

    def forward(self, template_list,search_list,template_anno_list,his_search,his_hp):
        zx,his_search,his_hp = self.forward_features(template_list,search_list,template_anno_list,his_search,his_hp)
        return zx,his_search,his_hp
