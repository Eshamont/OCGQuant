import math

import torch
import torch.nn as nn
from tqdm import tqdm

from .quantize import quantize_e2m1, quantize_ue4m3


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):
    def __init__(self, module, cpu_offload=False):
        super().__init__()
        self.module = module
        if hasattr(module, "attention_type"):
            self.attention_type = module.attention_type
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *args, **kwargs):
        if self.cpu_offload:
            self.input_args.append(to(args, device="cpu"))
            self.input_kwargs.append(to(kwargs, device="cpu"))
        else:
            self.input_args.append(args)
            self.input_kwargs.append(kwargs)
        raise ForwardInterrupt


def to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to(x, device) for x in obj)
    if isinstance(obj, list):
        return [to(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    return obj


def maybe_first_element(obj):
    if isinstance(obj, (tuple, list)):
        return obj[0]
    return obj


def build_calibration_data(model_path, dataset, seed, seqlen, samples, get_loaders):
    trainloader, _ = get_loaders(dataset, nsamples=samples, seed=seed, seqlen=seqlen, model=model_path)
    calibration_data = []
    for sample in trainloader:
        calibration_data.append(sample[0] if isinstance(sample, tuple) else sample)
    return calibration_data


def _quantize_weight_column(w_col, scale_col, quant_type):
    return quantize_e2m1(w_col / scale_col) * scale_col


def _get_weight_global_scale(w):
    global_scale = torch.max(w.abs()).float() / (448.0 * 6.0)
    if global_scale == 0:
        global_scale = torch.tensor(1e-9, device=w.device, dtype=torch.float32)
    return global_scale


def _get_weight_group_scales(w, quant_type, group_size, global_scale=None):
    if global_scale is None:
        global_scale = _get_weight_global_scale(w)
    w_for_scale = w.float() / global_scale

    pad = (group_size - w_for_scale.shape[1] % group_size) % group_size
    if pad:
        w_for_scale = torch.nn.functional.pad(w_for_scale, (0, pad))
    groups = w_for_scale.view(w_for_scale.shape[0], -1, group_size)
    scales = groups.abs().amax(dim=2) / 6.0
    scales[scales == 0] = 1e-9
    return quantize_ue4m3(scales) * global_scale


class OcgGPTQHandle:
    def __init__(self, layer, quant_type, quantization_order="default", block_size=128, rel_damp=0.1):
        self.layer = layer
        self.quant_type = quant_type
        self.quantization_order = quantization_order
        self.block_size = block_size
        self.rel_damp = rel_damp
        self.H = None
        self.num_samples = 0

    @torch.no_grad()
    def update(self, inp):
        batch_size = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).float()
        if self.H is None:
            self.H = torch.zeros((inp.shape[-1], inp.shape[-1]), device=inp.device, dtype=torch.float32)
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        inp.mul_(math.sqrt(alpha))
        self.H.addmm_(inp.T, inp)
        self.num_samples += batch_size

    @torch.no_grad()
    def quantize(self, reorder_index):
        if self.H is None:
            raise RuntimeError("GPTQ needs at least one calibration input.")

        orig_dtype = self.layer.weight.dtype
        device = self.layer.weight.device
        reorder_index = reorder_index.to(device=device, dtype=torch.long)
        w = self.layer.weight.detach().float()[:, reorder_index].contiguous()
        H = self.H[reorder_index][:, reorder_index].float()
        d_col = w.shape[1]
        group_size = 16
        num_groups = math.ceil(d_col / group_size)
        global_scale = _get_weight_global_scale(w)
        scales = torch.empty((w.shape[0], num_groups), device=device, dtype=torch.float32)

        base_group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[:d_col]
        if self.quantization_order == "activation":
            perm = torch.argsort(H.diag(), descending=True)
            group_idx = base_group_idx[perm]
            group_columns = [
                torch.nonzero(group_idx == group, as_tuple=False).flatten()
                for group in range(num_groups)
            ]
        else:
            perm = torch.arange(d_col, device=device)
            group_idx = base_group_idx
            group_columns = None
        perm_inv = torch.argsort(perm)

        H = H[perm][:, perm]
        w = w[:, perm]

        zero_cols = torch.nonzero(w.eq(0).all(dim=0), as_tuple=False).flatten()
        if zero_cols.numel() > 0:
            H[zero_cols, :] = 0
            H[:, zero_cols] = 0
            H[zero_cols, zero_cols] = 1

        damp = self.rel_damp * torch.diag(H).mean()
        H[range(d_col), range(d_col)] += damp
        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError:
            H_inv_cho = torch.eye(d_col, device=device, dtype=torch.float32)

        current_group = None
        current_scales = None
        group_scale_cache = [None] * num_groups if self.quantization_order == "activation" else None
        for c1 in range(0, d_col, self.block_size):
            c2 = min(c1 + self.block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()
            errs = torch.zeros_like(w_blk)
            H_blk = H_inv_cho[c1:c2, c1:c2]

            for i in range(ncols):
                w_ci = w_blk[:, i]
                d = H_blk[i, i]
                if self.quantization_order == "activation":
                    g_idx = group_idx[c1 + i]
                else:
                    g_idx = (c1 + i) // group_size
                if current_group != int(g_idx):
                    current_group = int(g_idx)
                    if self.quantization_order == "activation" and group_scale_cache[current_group] is not None:
                        current_scales = group_scale_cache[current_group]
                    else:
                        if self.quantization_order == "activation":
                            group_weight = w[:, group_columns[current_group]]
                        else:
                            group_start = current_group * group_size
                            group_end = min(group_start + group_size, d_col)
                            group_weight = w[:, group_start:group_end]
                        current_scales = _get_weight_group_scales(
                            group_weight,
                            self.quant_type,
                            group_size,
                            global_scale=global_scale,
                        )
                        current_scales = current_scales.reshape(w.shape[0], -1)[:, 0]
                        if self.quantization_order == "activation":
                            group_scale_cache[current_group] = current_scales
                    scales[:, current_group] = current_scales.float()
                w_q = _quantize_weight_column(w_ci, current_scales, self.quant_type)
                w[:, c1 + i] = w_q
                err = (w_ci - w_q) / d
                w_blk[:, i:].addr_(err, H_blk[i, i:], alpha=-1)
                errs[:, i] = err

            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        w = w[:, perm_inv].contiguous()
        new_weight = torch.empty_like(self.layer.weight.data, dtype=torch.float32)
        new_weight[:, reorder_index] = w
        self.layer.weight.data.copy_(new_weight.to(orig_dtype))


@torch.no_grad()
def ocg_gptq_quantization(model, calibration_data, reorder_index, quant_type, device, args):
    # print("OCGQuant quantization...")

    model.config.use_cache = False
    blocks = model.model.layers
    act_offload_device = "cpu" if args.cpu_offload_activations else device

    blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
    model.get_input_embeddings().to(device)
    blocks[0] = blocks[0].to(device)

    for sample in calibration_data:
        try:
            model(sample.to(device=device))
        except ForwardInterrupt:
            if args.cpu_offload_activations:
                sample = sample.cpu()

    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    model.get_input_embeddings().cpu()

    for block_idx, block in enumerate(tqdm(blocks, desc="gptq blocks")):
        block = block.to(device)

        handles = {}
        hooks = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, nn.Linear):
                key = f"layers.{block_idx}.{layer_name}.input"
                if key not in reorder_index:
                    continue
                handles[layer_name] = OcgGPTQHandle(
                    layer,
                    quant_type=quant_type,
                    quantization_order=args.quantization_order,
                    rel_damp=args.rel_damp,
                )

                def update_handle_hook(name):
                    def _hook(_, inp, _out):
                        handles[name].update(inp[0])
                    return _hook

                hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

        for hook in hooks.values():
            hook.remove()

        for layer_name, handle in handles.items():
            key = f"layers.{block_idx}.{layer_name}.input"
            handle.quantize(reorder_index[key])

        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).detach().to(act_offload_device)
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"] = out
            else:
                raise ValueError("Unsupported block input format.")

        blocks[block_idx] = block.cpu()
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return model
