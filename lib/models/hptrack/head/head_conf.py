import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
#from lib.models.hptrack.PreciseRoIPooling.pytorch.prroi_pool.prroi_pool import PrRoIPool2D

class ScoreDecoder(nn.Module):
    def __init__(self,num_heads=8,hidden_dim=512, pool_size=4,dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.search_prroipool = PrRoIPool2D(pool_size,pool_size,spatial_scale=1.0)
        self.score_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)for _ in range(1)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)for _ in range(1)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)for _ in range(1)])
        self.out_proj=nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)for _ in range(1)])
        self.norm_q = nn.ModuleList([nn.LayerNorm(hidden_dim)for _ in range(1)])
        self.attn_drop = nn.Dropout(dropout)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid())
        self._init_weights()
    def _init_weights(self):
        trunc_normal_(self.score_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def cross_attention(self, x, memory, layer_id):
        shortcut = x
        x_norm = self.norm_q[layer_id](x)
        q = rearrange(self.q_proj[layer_id](x_norm),'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.k_proj[layer_id](memory),'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.v_proj[layer_id](memory),'b t (h d) -> b h t d', h=self.num_heads)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out,'b h t d -> b t (h d)')
        out = self.out_proj[layer_id](out)
        x = shortcut + out
        return x

    def forward(self,query,search_feat,search_box):
        B, C, H, W = search_feat.shape
        # ROI pooling
        search_box = search_box.clone() * W
        batch_index = torch.arange(B,dtype=torch.float32,device=search_box.device).view(-1, 1)
        target_roi = torch.cat([batch_index, search_box.view(-1, 4)],dim=1)
        search_roi_feat = self.search_prroipool(search_feat,target_roi)
        search_roi_feat = rearrange(search_roi_feat,'b c h w -> b (h w) c')
        x = self.score_token.expand(B, -1, -1)
        x = self.cross_attention( x,search_roi_feat,0)
        x = query[:, :1, :] + x
        out_scores = self.score_head(x)

        return out_scores


class ConfidenceHead(nn.Module):
    def __init__(self, dim, dropout=0.1):

        super().__init__()
        self.fc1 = nn.Linear(dim, dim//4)
        self.act = nn.GELU()  # ReLU
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim//4, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self,query=None, search_feat=None, search_box=None):
        cls_feat = query[:, :1, :]
        #hp_token = hp[:, 1:, :]
        if cls_feat.dim() == 3:
            B, N, D = cls_feat.shape
            x = cls_feat.view(B * N, D)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(B, N, 1)
        else:  # (B, D)
            x = self.fc1(cls_feat)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
        return self.sigmoid(x)


