import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from .quantize import *

KERNEL_BUILD_DIR = Path(__file__).resolve().parents[1] / "kernels" / "build"
sys.path.insert(0, str(KERNEL_BUILD_DIR))
try:
    import ocgcuda
except ImportError:
    ocgcuda = None

import math
import random


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def NVFP4_reorder_quantize_w(w, reorder_index, select_num):
    if use_nvfp4_kernel_path():
        scale = torch.max(w.abs()).float() / (448.0*6.0)
        scale = torch.where(scale == 0, 1.0, scale)
        qw, scale_w = ocgcuda.reorder_quantize_w(w/scale, reorder_index, select_num)
        return qw, scale_w, scale

    index = reorder_index.to(device=w.device, dtype=torch.long)
    w_reordered = torch.index_select(w, 1, index)
    identity = torch.arange(w_reordered.shape[1], device=w.device)
    return fake_reorder_quantize_w(
        w_reordered,
        identity,
        select_num,
        dtype='NVFP4',
    )
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        select_num, 
        reorder_index,
        out_reorder_index=None,
        quant_type='NVFP4',
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features

        weight = originalLayer.weight.data
        bias = originalLayer.bias
        if out_reorder_index is not None:
            out_index = out_reorder_index.to(device=weight.device, dtype=torch.long)
            weight = torch.index_select(weight, 0, out_index).contiguous()
            if bias is not None:
                bias = torch.index_select(bias, 0, out_index).contiguous()

        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        self.select_num = select_num
        self.quant_type = quant_type
        self.track_act_global_scale = False
        self.register_buffer('act_global_scale', torch.zeros((), dtype=torch.float32))

        W, scale_w, scale = NVFP4_reorder_quantize_w(weight, reorder_index.to(torch.int16).cuda(), select_num)
        self.register_buffer('W', W)
        self.register_buffer('scale_w', scale_w)
        self.register_buffer('scale', scale)
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        qx, scale_x, scale, bsz, q_len = x

        if use_nvfp4_kernel_path():
            matmul_scale = scale * self.scale
            if torch.is_tensor(matmul_scale):
                matmul_scale = float(matmul_scale.detach().float().cpu())
            y = ocgcuda.matmul(qx, self.W, scale_x, self.scale_w, matmul_scale)
        else:
            y = F.linear(qx, self.W)
        
        torch.cuda.synchronize()
        if self.bias is not None:
            y = y + self.bias

        if bsz is not None:
            y = y.reshape(bsz, q_len, -1)
        else:
            y = y.reshape(q_len, -1)
        return y


def set_act_global_scale_tracking(module, enabled, reset=False):
    for child in module.modules():
        if isinstance(child, QLinearLayer):
            if reset:
                child.act_global_scale.zero_()
            child.track_act_global_scale = enabled


def get_act_global_scale_summary(module):
    scales = [
        child.act_global_scale.detach().float().cpu()
        for child in module.modules()
        if isinstance(child, QLinearLayer) and child.quant_type == "NVFP4"
    ]
    if not scales:
        return None
    stacked = torch.stack(scales)
    calibrated = stacked[stacked > 0]
    if calibrated.numel() == 0:
        return {"count": len(scales), "calibrated": 0, "min": 0.0, "max": 0.0}
    return {
        "count": len(scales),
        "calibrated": int(calibrated.numel()),
        "min": float(calibrated.min()),
        "max": float(calibrated.max()),
    }

    
