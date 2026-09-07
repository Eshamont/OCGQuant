import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from .qLinearLayer import find_qlinear_layers
from .qLlamaLayer import QLlamaDecoderLayer
from .qQwen3Layer import QQwen3DecoderLayer


from functools import partial

import math


def reorder_model_llama(model, device, kv_cache, reorder_index, select_nums, quant_type, fuse_rmsnorm_reorder=False, fuse_rmsnorm_reorder_kernel=False, r5_reorder_absorb=False):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], LlamaDecoderLayer):
            m = QLlamaDecoderLayer(
                originalLayer=layers[i],
                kv_cache=kv_cache,
                select_nums=select_nums,
                reorder_index=reorder_index,
                layer_idx=i,
                quant_type=quant_type,
                fuse_rmsnorm_reorder=fuse_rmsnorm_reorder,
                fuse_rmsnorm_reorder_kernel=fuse_rmsnorm_reorder_kernel,
                r5_reorder_absorb=r5_reorder_absorb,
            )
        elif isinstance(layers[i], QLlamaDecoderLayer):
            m = layers[i]
            
        nameTemplate = 'layers.{}.{}.{}.{}'
        m.mlp.register_buffer('up_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')].to(torch.int16))
        m.mlp.register_buffer('down_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('q_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('o_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')].to(torch.int16))
        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def reorder_model_qwen3(model, device, kv_cache, reorder_index, select_nums, quant_type, fuse_rmsnorm_reorder=False, fuse_rmsnorm_reorder_kernel=False, r5_reorder_absorb=False):
    model.config.use_cache = False
    layers = model.model.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], Qwen3DecoderLayer):
            m = QQwen3DecoderLayer(
                originalLayer=layers[i],
                kv_cache=kv_cache,
                select_nums=select_nums,
                reorder_index=reorder_index,
                layer_idx=i,
                quant_type=quant_type,
                fuse_rmsnorm_reorder=fuse_rmsnorm_reorder,
                fuse_rmsnorm_reorder_kernel=fuse_rmsnorm_reorder_kernel,
                r5_reorder_absorb=r5_reorder_absorb,
            )
        elif isinstance(layers[i], QQwen3DecoderLayer):
            m = layers[i]
        else:
            raise TypeError(
                f"Unsupported Qwen3 decoder layer type: {type(layers[i]).__name__}. "
                "QQwen3DecoderLayer only supports Qwen3."
            )

        nameTemplate = 'layers.{}.{}.{}.{}'
        m.mlp.register_buffer('up_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')].to(torch.int16))
        m.mlp.register_buffer('down_reorder_index', reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('q_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')].to(torch.int16))
        m.self_attn.register_buffer('o_reorder_index', reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')].to(torch.int16))
        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model
