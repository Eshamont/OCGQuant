import torch
import torch.nn.functional as F
import numpy as np
import gc
import os
import weakref
import math
import random
import sys
from pathlib import Path

KERNEL_BUILD_DIR = Path(__file__).resolve().parents[1] / "kernels" / "build"
sys.path.insert(0, str(KERNEL_BUILD_DIR))
try:
    import ocgcuda
except ImportError:
    ocgcuda = None


def use_nvfp4_kernel_path(device=None):
    return (
        ocgcuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == (12, 0)
    )


def use_reorder_metadata_fast_path():
    return os.environ.get("ARC_DISABLE_REORDER_META", "0").lower() not in {"1", "true", "yes"}


def call_rmsnorm_quantize_x(ocgcuda_module, hidden_states, weight, eps, reorder_index, select_num, global_scale=1.0):
    return ocgcuda_module.rmsnorm_quantize_x(
        hidden_states, weight, eps, reorder_index, select_num, global_scale
    )


_REORDER_GROUP_METADATA_CACHE = {}


def get_reorder_group_metadata(reorder_index, group_size):
    if reorder_index is None:
        return None, None
    key = (id(reorder_index), group_size)
    # Inference tensors have no version counter, so their metadata is not cached.
    version = None if reorder_index.is_inference() else reorder_index._version
    cached = _REORDER_GROUP_METADATA_CACHE.get(key)
    if version is not None and cached is not None and cached[0]() is reorder_index and cached[1] == version:
        return cached[2]

    idx = reorder_index.contiguous()
    groups = idx.view(-1, group_size)
    # Own the storage even for a single group; a view would keep the index alive.
    base = groups[:, 0].to(dtype=torch.int16, copy=True, memory_format=torch.contiguous_format)
    offsets = torch.arange(group_size, device=idx.device, dtype=torch.int32)
    contiguous = (groups.to(torch.int32) == (base.to(torch.int32).unsqueeze(1) + offsets)).all(dim=1).to(torch.uint8).contiguous()
    metadata = (base, contiguous)

    if version is not None:
        def remove_cache(ref):
            entry = _REORDER_GROUP_METADATA_CACHE.get(key)
            if entry is not None and entry[0] is ref:
                del _REORDER_GROUP_METADATA_CACHE[key]

        _REORDER_GROUP_METADATA_CACHE[key] = (weakref.ref(reorder_index, remove_cache), version, metadata)
    return metadata


def get_reorder_group_size(hidden_dim):
    return 32 if hidden_dim in {3584, 17408, 18944} else 16


def get_nvfp4_activation_scale(x, scale_owner=None):
    scale = torch.max(x.abs()).to(torch.float32) / (448.0 * 6.0)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)

    if scale_owner is None or not hasattr(scale_owner, "act_global_scale"):
        return scale

    if getattr(scale_owner, "track_act_global_scale", False):
        scale_owner.act_global_scale.copy_(
            torch.maximum(scale_owner.act_global_scale.to(scale.device), scale.detach()).to(
                scale_owner.act_global_scale.device
            )
        )

    stored_scale = scale_owner.act_global_scale.to(scale.device)
    return torch.where(stored_scale > 0, stored_scale, scale)


def quantize_e2m1(tensor):
    # Nearest-value E2M1 quantization without materializing a 16x candidate tensor.
    abs_tensor = tensor.abs()
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    positive_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # Match the old argmin tie-break: positive ties choose the lower value,
    # negative ties choose the more negative value because candidates were sorted.
    pos_indices = torch.bucketize(abs_tensor.contiguous(), thresholds, right=False)
    neg_indices = torch.bucketize(abs_tensor.contiguous(), thresholds, right=True)
    indices = torch.where(tensor < 0, neg_indices, pos_indices)
    quantized_abs = positive_vals[indices]
    quantized_abs.copysign_(tensor)
    return quantized_abs

def dequantize_e2m1(tensor):
    return tensor

def quantize_ue4m3(tensor):
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    
    reconstructed_val = (1 + quantized_mantissa_val) * (2**exponent)
    return reconstructed_val

def dequantize_ue4m3(tensor):
    return tensor

def quantize_ue8m0(tensor):
    exponent = torch.ceil(torch.log2(tensor + 1e-9))
    exponent = torch.clamp(exponent, min=-127, max=127)
    
    reconstructed_val = (2**exponent)
    return reconstructed_val

def dequantize_ue8m0(tensor):
    return tensor


def quantize_nvfp4_tensor(tensor, group_size=16):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 6.0
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue4m3(scale)
    dequantized_scale = dequantize_ue4m3(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e2m1_tensor = quantize_e2m1(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e2m1(quantized_e2m1_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def get_e3m2_values(device, dtype):
    vals =[0.0]
    vals.extend([0.0625, 0.125, 0.1875])
    
    mantissas =[1.0, 1.25, 1.5, 1.75]
    for E in range(1, 8): 
        exponent_val = 2 ** (E - 3)
        for m in mantissas:
            vals.append(m * exponent_val)
            
    pos_vals = torch.tensor(vals, device=device, dtype=dtype)
    all_vals = torch.cat([-pos_vals, pos_vals]).unique()
    return torch.sort(all_vals)[0]

def quantize_e3m2(tensor):
    representable_vals = get_e3m2_values(tensor.device, tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    
    return representable_vals[indices]

def dequantize_e3m2(tensor):
    return tensor

def quantize_mxfp6_tensor(tensor, group_size=32):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    
    scale = max_abs_val / 28.0 
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e3m2_tensor = quantize_e3m2(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e3m2(quantized_e3m2_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)


def fake_reorder_quantize_w(w, reorder_index, select_num, dtype='NVFP4', group_size=16):
    orig_dtype = w.dtype
    scale = torch.max(w.abs()).to(torch.float32) / (448.0*6.0)
    scale = torch.where(scale == 0, 1.0, scale)
    quantize_func = lambda t: quantize_nvfp4_tensor(t, group_size=group_size)

    w_fp32 = w.to(torch.float32) / scale
    scale_w = w_fp32.abs().max(dim=1, keepdim=True)[0]
    
    if select_num == 0:
        q_w = quantize_func(w_fp32) * scale
        return q_w.to(orig_dtype), scale_w.to(orig_dtype), scale.to(orig_dtype)
    else:
        topk_index = reorder_index[-select_num:]
        q_w = torch.cat([quantize_func(w_fp32), quantize_func(w_fp32[:, topk_index])], dim=1) * scale
        return q_w.to(orig_dtype), scale_w.to(orig_dtype), scale.to(orig_dtype)

def fake_reorder_quantize_x(x, reorder_index, select_num, dtype='NVFP4', scale_owner=None, group_size=16):
    orig_dtype = x.dtype
    scale = get_nvfp4_activation_scale(x, scale_owner)
    quantize_func = lambda t: quantize_nvfp4_tensor(t, group_size=group_size)

    x_fp32 = x.to(torch.float32) / scale
    scale_x = x_fp32.abs().max(dim=1, keepdim=True)[0]
    
    if select_num == 0:
        q_x = quantize_func(x_fp32) * scale
        return q_x.to(orig_dtype), scale_x.to(orig_dtype), scale.to(orig_dtype)
    else:
        topk_index = reorder_index[-select_num:]
        q_x = quantize_func(x_fp32)
        error_e = x_fp32 - q_x
        q_error_k = quantize_func(error_e[:, topk_index])
        
        ret_x = torch.cat([q_x, q_error_k], dim=1) * scale
        return ret_x.to(orig_dtype), scale_x.to(orig_dtype), scale.to(orig_dtype)

@torch.no_grad()
def hadamard_transform(x, normalize=True, block_size=-1):
    n = x.shape[-1]
    if block_size == -1:
        if n <= 0 or (n & (n - 1)) != 0:
            return x
    else:
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size {block_size}")
        if n % block_size != 0:
            raise ValueError(f" {n} block_size {block_size}")
    
    original_shape = x.shape
    
    if block_size != -1:
        num_blocks = n // block_size
        x = x.view(-1, num_blocks, block_size)
        batch_dim = x.shape[0]
        current_n = block_size
    else:
        x = x.view(-1, n)
        batch_dim = x.shape[0]
        current_n = n
    
    h = x.clone()
    num_stages = int(torch.log2(torch.tensor(current_n, dtype=torch.float32)).item())
    
    for stage in range(num_stages):
        stage_block_size = 2 ** (stage + 1)
        half_block_size = stage_block_size // 2
        if block_size != -1:
            temp = h.view(batch_dim, -1, stage_block_size)
        else:
            temp = h.view(batch_dim, -1, stage_block_size)
        front_half = temp[:, :, :half_block_size]
        back_half = temp[:, :, half_block_size:]
        new_front = front_half + back_half
        new_back = front_half - back_half
        h = torch.cat([new_front, new_back], dim=2)
        if block_size != -1:
            h = h.view(batch_dim, -1, current_n)
        else:
            h = h.view(batch_dim, current_n)
    
    if normalize:
        h = h / torch.sqrt(torch.tensor(current_n, dtype=torch.float32))
    if block_size != -1:
        h = h.view(-1, num_blocks * block_size)
    h = h.view(original_shape)
    return h
