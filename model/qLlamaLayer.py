import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from .qLinearLayer import QLinearLayer
from .quantize import *
import sys
from pathlib import Path

KERNEL_BUILD_DIR = Path(__file__).resolve().parents[1] / "kernels" / "build"
sys.path.insert(0, str(KERNEL_BUILD_DIR))
try:
    import ocgcuda
except ImportError:
    ocgcuda = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RMSNORM_FUSION_HIDDEN_DIMS = {2048, 3072, 3584, 4096, 5120, 8192}


def _is_quantized_linear_input(x):
    return isinstance(x, tuple)




def _get_rmsnorm_eps(original_norm):
    return float(getattr(original_norm, "variance_epsilon", getattr(original_norm, "eps", 1e-6)))


def _get_frozen_nvfp4_scale(scale_owner):
    if scale_owner is None or not hasattr(scale_owner, "act_global_scale"):
        return None
    scale = scale_owner.act_global_scale.detach().float()
    if scale.numel() != 1 or float(scale.cpu()) <= 0:
        return None
    return float(scale.cpu())


def _fused_rmsnorm_reorder_quantize_x(
    hidden_states,
    original_norm,
    reorder_index,
    select_num,
    quant_type,
    scale_owner,
    use_cuda_kernel=False,
):
    bsz, q_len, _ = hidden_states.shape
    if use_cuda_kernel:
        global_scale = _get_frozen_nvfp4_scale(scale_owner)
        if (
            global_scale is not None
            and quant_type == 'NVFP4'
            and use_nvfp4_kernel_path()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        ):
            flat_hidden_states = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
            # Fused RMSNorm kernels process 16 elements per thread.
            reorder_base, reorder_contiguous = get_reorder_group_metadata(reorder_index, 16)
            if use_reorder_metadata_fast_path() and hasattr(ocgcuda, "rmsnorm_quantize_x_meta"):
                qx, scale_x = ocgcuda.rmsnorm_quantize_x_meta(
                    flat_hidden_states,
                    original_norm.weight.contiguous(),
                    _get_rmsnorm_eps(original_norm),
                    reorder_index,
                    reorder_base,
                    reorder_contiguous,
                    select_num,
                    global_scale,
                )
            else:
                qx, scale_x = call_rmsnorm_quantize_x(
                    ocgcuda,
                    flat_hidden_states,
                    original_norm.weight.contiguous(),
                    _get_rmsnorm_eps(original_norm),
                    reorder_index,
                    select_num,
                    global_scale,
                )
            return qx, scale_x, global_scale, bsz, q_len

    hidden_states = original_norm(hidden_states).reshape(bsz * q_len, -1).contiguous().detach()
    qx, scale_x, scale = reorder_quantize_x(hidden_states, reorder_index, select_num, quant_type, scale_owner)
    return qx, scale_x, scale, bsz, q_len


@torch.no_grad()
def quantize_int_group(w, nbits, group_size):
    savedShape = w.shape
    w = w.reshape(-1, group_size)
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    q_max = (2**(nbits)-1)
    q_min = (0)
    scales = (w_max-w_min).clamp(min=1e-5) / q_max
    base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    return w.reshape(savedShape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_rope_theta(attn_module):
    theta = getattr(attn_module, "rope_theta", None)
    if theta is not None:
        return theta
    config = attn_module.config
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return theta
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    return rope_scaling.get("rope_theta", 10000.0)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def NVFP4_reorder_quantize_x(x, reorder_index, select_num, scale_owner=None):
    if not use_nvfp4_kernel_path():
        index = reorder_index.to(device=x.device, dtype=torch.long)
        identity = torch.arange(x.shape[-1], device=x.device)
        return fake_reorder_quantize_x(
            torch.index_select(x, 1, index),
            identity,
            select_num,
            dtype='NVFP4',
            scale_owner=scale_owner,
        )

    scale = get_nvfp4_activation_scale(x, scale_owner)
    group_size = get_reorder_group_size(x.shape[-1])
    reorder_base, reorder_contiguous = get_reorder_group_metadata(reorder_index, group_size)
    if use_reorder_metadata_fast_path() and hasattr(ocgcuda, "reorder_quantize_x_meta"):
        qx, scale_x = ocgcuda.reorder_quantize_x_meta(x/scale, reorder_index, reorder_base, reorder_contiguous, select_num)
    else:
        qx, scale_x = ocgcuda.reorder_quantize_x(x/scale, reorder_index, select_num)
    return qx, scale_x, scale

def reorder_quantize_x(x, reorder_index, select_num, quant_type='NVFP4', scale_owner=None):
    return NVFP4_reorder_quantize_x(x, reorder_index, select_num, scale_owner)

class QLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: LlamaDecoderLayer,
        kv_cache,
        select_nums,
        reorder_index,
        layer_idx,
        quant_type,
        fuse_rmsnorm_reorder=False,
        fuse_rmsnorm_reorder_kernel=False,
        r5_reorder_absorb=False,
    ):
        super().__init__()
       
        self.hidden_size = getattr(originalLayer, "hidden_size", originalLayer.self_attn.config.hidden_size)
        self.self_attn = QLlamaAttention(
            originalLayer.self_attn,
            kv_cache,
            select_nums=select_nums,
            reorder_index=reorder_index,
            i=layer_idx,
            quant_type=quant_type,
            r5_reorder_absorb=r5_reorder_absorb,
        )
        # self.self_attn = originalLayer.self_attn
        self.mlp = QLlamaMLP(
            originalLayer.mlp,
            select_nums=select_nums,
            reorder_index=reorder_index,
            i=layer_idx,
            quant_type=quant_type,
            r5_reorder_absorb=r5_reorder_absorb,
        )
        # self.mlp = originalLayer.mlp
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.input_layernorm = QLlamaRMSNorm(
            originalLayer.input_layernorm,
            reorder_index=reorder_index[nameTemplate.format(layer_idx, 'self_attn', 'q_proj', 'input')],
            select_num=select_nums[nameTemplate.format(layer_idx, 'self_attn', 'q_proj', 'input')],
            quant_type=quant_type,
            fuse_rmsnorm_reorder=fuse_rmsnorm_reorder,
            fuse_rmsnorm_reorder_kernel=fuse_rmsnorm_reorder_kernel,
            scale_owner=self.self_attn.q_proj,
        )
        self.post_attention_layernorm = QLlamaRMSNorm(
            originalLayer.post_attention_layernorm,
            reorder_index=reorder_index[nameTemplate.format(layer_idx, 'mlp', 'up_proj', 'input')],
            select_num=select_nums[nameTemplate.format(layer_idx, 'mlp', 'up_proj', 'input')],
            quant_type=quant_type,
            fuse_rmsnorm_reorder=fuse_rmsnorm_reorder,
            fuse_rmsnorm_reorder_kernel=fuse_rmsnorm_reorder_kernel,
            scale_owner=self.mlp.up_proj,
        )

    def to(self, *args, **kwargs):
        super(QLlamaDecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if past_key_value is None:
            past_key_value = past_key_values

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
   
        
class QLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: LlamaRMSNorm,
        reorder_index=None,
        select_num=0,
        quant_type='NVFP4',
        fuse_rmsnorm_reorder=False,
        fuse_rmsnorm_reorder_kernel=False,
        scale_owner=None,
    ):
        super().__init__()
        self.originalNorm = originalNorm
        self.select_num = select_num
        self.quant_type = quant_type
        self.scale_owner = scale_owner
        self.fuse_rmsnorm_reorder_kernel = fuse_rmsnorm_reorder_kernel
        self.fuse_rmsnorm_reorder = (
            (fuse_rmsnorm_reorder or fuse_rmsnorm_reorder_kernel)
            and quant_type == 'NVFP4'
            and originalNorm.weight.numel() in RMSNORM_FUSION_HIDDEN_DIMS
            and reorder_index is not None
            and scale_owner is not None
        )
        if reorder_index is not None:
            self.register_buffer('reorder_index', reorder_index.to(torch.int16))
        else:
            self.reorder_index = None

    @torch.no_grad()
    def forward(self, hidden_states):
        if self.fuse_rmsnorm_reorder:
            result = _fused_rmsnorm_reorder_quantize_x(
                hidden_states,
                self.originalNorm,
                self.reorder_index,
                self.select_num,
                self.quant_type,
                self.scale_owner,
                use_cuda_kernel=self.fuse_rmsnorm_reorder_kernel,
            )
            if result is not None:
                return result
        result = self.originalNorm(hidden_states)
            
#         if self.args.abits < 16:
#             result = self.act_quant(result)
        
        
        return result
    
    def to(self, *args, **kwargs):
        super(QLlamaRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
       
        return self

class QLlamaAttention(nn.Module):

    def __init__(
        self, 
        originalAttn: LlamaAttention,
        kv_cache,
        select_nums,
        reorder_index,
        i,
        quant_type,
        r5_reorder_absorb=False,
    ):
        super().__init__()
        self.q_kv_cache = kv_cache
        self.config = originalAttn.config
        self.hidden_size = getattr(originalAttn, "hidden_size", self.config.hidden_size)
        self.num_heads = getattr(originalAttn, "num_heads", self.config.num_attention_heads)
        self.head_dim = getattr(originalAttn, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(originalAttn, "num_key_value_heads", self.config.num_key_value_heads)
        self.num_key_value_groups = getattr(
            originalAttn,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.max_position_embeddings = getattr(
            originalAttn,
            "max_position_embeddings",
            self.config.max_position_embeddings,
        )
        self.rope_theta = get_rope_theta(originalAttn)
        self.layer_idx = i
        self.quant_type = quant_type
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        nameTemplate = 'layers.{}.{}.{}.{}'
        o_input_reorder_index = reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')]
        self.q_proj = QLinearLayer(
            originalAttn.q_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            quant_type=quant_type
        )
        self.k_proj = QLinearLayer(
            originalAttn.k_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            quant_type=quant_type
        )
        self.v_proj = QLinearLayer(
            originalAttn.v_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            quant_type=quant_type
        )
        self.o_proj = QLinearLayer(
            originalAttn.o_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            reorder_index=o_input_reorder_index,
            quant_type=quant_type
        )
        self.rotary_emb = getattr(originalAttn, "rotary_emb", None)


        self.attention_dropout = getattr(originalAttn, "attention_dropout", self.config.attention_dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QLlamaAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        if self.rotary_emb is not None:
            self.rotary_emb = self.rotary_emb.to(*args, **kwargs)
      
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        if _is_quantized_linear_input(hidden_states):
            qx, scale_x, scale, bsz, q_len = hidden_states
            hidden_states = (qx, scale_x, scale, bsz, q_len)
        else:
            bsz, q_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(bsz*q_len, -1).contiguous().detach()
            qx, scale_x, scale = reorder_quantize_x(hidden_states, self.q_reorder_index, self.q_proj.select_num, self.quant_type, self.q_proj)
            torch.cuda.synchronize()
            hidden_states = (qx, scale_x, scale, bsz, q_len)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        
        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=64)
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_embeddings is None:
            if self.rotary_emb is None:
                raise ValueError("position_embeddings is required when the attention module has no rotary_emb.")
            cos, sin = self.rotary_emb(value_states, position_ids)
         
        else:
            cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            
        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=64)
            
            
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Quantize the attention output
      
        attn_output = attn_output.reshape(bsz*q_len, -1).contiguous().detach()


        o_activation_index = self.o_reorder_index.to(device=attn_output.device)
        qx, scale_x, scale = reorder_quantize_x(
            attn_output,
            o_activation_index,
            self.o_proj.select_num,
            self.quant_type,
            self.o_proj,
        )
        torch.cuda.synchronize()
        attn_output = (qx, scale_x, scale, bsz, q_len)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    

class QLlamaMLP(nn.Module):
    def __init__(
        self,
        originalMLP: LlamaMLP,
        select_nums,
        reorder_index,
        i,
        quant_type,
        r5_reorder_absorb=False,
    ):
        super().__init__()
        nameTemplate = 'layers.{}.{}.{}.{}'

        self.quant_type = quant_type
        self.r5_reorder_absorb = r5_reorder_absorb
        
        self.gate_proj = QLinearLayer(
            originalMLP.gate_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            out_reorder_index=(reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')] if self.r5_reorder_absorb else None),
            quant_type=self.quant_type
        )
        self.down_proj = QLinearLayer(
            originalMLP.down_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            quant_type=self.quant_type
        )
        self.up_proj = QLinearLayer(
            originalMLP.up_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            out_reorder_index=(reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')] if self.r5_reorder_absorb else None),
            quant_type=self.quant_type
        )
        self.act_fn = originalMLP.act_fn
        self.layer_idx = i
        self.register_buffer(
            'down_identity_index',
            torch.arange(originalMLP.down_proj.in_features, dtype=torch.int16),
        )
        
        
    def to(self, *args, **kwargs):
        super(QLlamaMLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        

        return self

    @torch.no_grad()
    def forward(self, x):
        # input X: [b, seq, dim]: quantized

        if _is_quantized_linear_input(x):
            qx, scale_x, scale, bsz, q_len = x
            x = (qx, scale_x, scale, bsz, q_len)
        else:
            bsz, q_len, _ = x.shape
            x = x.reshape(bsz*q_len, -1).contiguous().detach()

            qx, scale_x, scale = reorder_quantize_x(x, self.up_reorder_index, self.up_proj.select_num, self.quant_type, self.up_proj)
            torch.cuda.synchronize()
            x = (qx, scale_x, scale, bsz, q_len)
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Quantize the activations and feed into down_proj

        bsz, q_len, _ = tmpResult.shape
        tmpResult = tmpResult.reshape(bsz*q_len, -1).contiguous().detach()
        

        down_activation_index = (
            self.down_identity_index.to(device=tmpResult.device)
            if self.r5_reorder_absorb
            else self.down_reorder_index.to(device=tmpResult.device)
        )
        qx, scale_x, scale = reorder_quantize_x(
            tmpResult,
            down_activation_index,
            self.down_proj.select_num,
            self.quant_type,
            self.down_proj,
        )
        torch.cuda.synchronize()
        tmpResult = (qx, scale_x, scale, bsz, q_len)

        return self.down_proj(tmpResult)
