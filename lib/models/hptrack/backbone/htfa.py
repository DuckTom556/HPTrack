import torch
import torch.nn.functional as F
import math
from torch import nn
import torch.utils.checkpoint as checkpoint
from torch.ao.nn.quantized import LayerNorm


class ResidualBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor,grad_ckpt):
        super().__init__()

        self.grad_ckpt = grad_ckpt
        self.mixer = SSMBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h):
        x = self.norm(x)
        output, h = checkpoint.checkpoint(self.mixer,x,h,use_reentrant=False) if self.grad_ckpt else self.mixer(x, h)
        output = output + x
        return output, h
class SSMBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()
        self.dt_scale = dt_scale
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, bias=conv_bias,
                                groups=self.d_inner,
                                padding=(d_conv - 1)//2)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    def forward(self, x, h):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x_cache = x.permute(0,2,1)
        x = self.conv1d( x_cache).permute(0,2,1)
        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        return output, h

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * (x.unsqueeze(-1))
        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.d_inner, self.d_state, device=deltaA.device)
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class Mamba(nn.Module):
    def __init__(self, d_model=512, d_inner=1024, bias=False, dt_rank=32, d_state=16, d_conv=3,dt_min=0.001,
                 dt_max=0.1, dt_init='random', dt_scale=1.0, conv_bias=True, dt_init_floor=0.0001,grad_ckpt=False):
        super().__init__()
        self.mamba = ResidualBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,
                                 dt_init,dt_max,dt_min,dt_init_floor,grad_ckpt)
    def forward(self, z):
        num=z.shape[1]
        his=None
        z0=z[:,0,:,:]
        for i in range(num):
            z_his=z[:,i,:,:]
            z_his,his=self.mamba(z_his,his)
        z=torch.cat([z0,z_his],dim=1)
        return z

def build_htfa(d_model,d_state,grad_ckpt):
    neck = Mamba(d_model=d_model,
                d_inner=2*d_model,
                dt_rank=d_model//16,
                d_state=d_state,
               grad_ckpt=grad_ckpt)
    return neck