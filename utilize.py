from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import gc
import torch
import functools
import random
import math
from tqdm import tqdm


SUPPORTED_REORDER_METRICS = {
    "max",
    "rms",
}


def _is_permutation(index: torch.Tensor) -> bool:
    if not torch.is_tensor(index) or index.dim() != 1 or index.dtype.is_floating_point:
        return False
    expected = torch.arange(index.numel(), device=index.device, dtype=index.dtype)
    return torch.equal(torch.sort(index).values, expected)


@torch.no_grad()
def get_reorder_index(model, act_scales, metric="max", group_size=16, select_nums=None, pack_mode="compact", outlier_masks=None, order_score_key="outlier", companion_score_key="companion"):
    if metric not in SUPPORTED_REORDER_METRICS:
        raise ValueError(f"Unsupported reorder metric: {metric}. Supported: {sorted(SUPPORTED_REORDER_METRICS)}")

    def plain_order(scores):
        _, sorted_index = torch.sort(scores, descending=False)
        assert _is_permutation(sorted_index)
        return sorted_index

    def packed_order(outlier_scores, companion_scores, select_num, outlier_mask=None):
        assert outlier_scores.dim() == 1
        assert companion_scores.dim() == 1
        assert outlier_scores.numel() == companion_scores.numel()

        identity = torch.arange(outlier_scores.numel(), device=outlier_scores.device, dtype=torch.long)
        if group_size <= 1 or select_num is None:
            assert _is_permutation(identity)
            return identity

        max_group_count = outlier_scores.numel() // group_size
        num_outliers = min(int(select_num), max_group_count)
        if num_outliers <= 0:
            assert _is_permutation(identity)
            return identity

        outlier_order = torch.argsort(outlier_scores, descending=True).tolist()
        if outlier_mask is not None:
            mask_cpu = outlier_mask.detach().bool().cpu()
            outlier_order = [idx for idx in outlier_order if bool(mask_cpu[idx].item())]
        outlier_pool = outlier_order[:num_outliers]
        if not outlier_pool:
            assert _is_permutation(identity)
            return identity
        outlier_set = set(outlier_pool)
        available = set(range(outlier_scores.numel())) - outlier_set
        companion_order = torch.argsort(companion_scores, descending=False).tolist()

        packed_groups = []
        used = set()
        for outlier in outlier_pool:
            companions = []
            for candidate in companion_order:
                if candidate in available:
                    companions.append(candidate)
                    available.remove(candidate)
                    if len(companions) == group_size - 1:
                        break
            if len(companions) < group_size - 1:
                break

            used.add(outlier)
            used.update(companions)
            packed_groups.append(torch.tensor(companions + [outlier], device=outlier_scores.device, dtype=torch.long))

        if not packed_groups:
            assert _is_permutation(identity)
            return identity

        if pack_mode == "swap_tail":
            packed_flat = torch.cat(packed_groups, dim=0)
            tail_len = packed_flat.numel()
            tail_start = outlier_scores.numel() - tail_len
            tail_positions = list(range(tail_start, outlier_scores.numel()))
            used_set = set(int(x) for x in used)
            tail_spares = [idx for idx in tail_positions if idx not in used_set]

            result = torch.empty_like(identity)
            spare_idx = 0
            for pos in range(tail_start):
                if pos in used_set:
                    result[pos] = tail_spares[spare_idx]
                    spare_idx += 1
                else:
                    result[pos] = pos
            result[tail_start:] = packed_flat
            assert spare_idx == len(tail_spares)
            assert _is_permutation(result)
            return result

        middle = torch.tensor(
            [idx for idx in identity.tolist() if idx not in used],
            device=outlier_scores.device,
            dtype=torch.long,
        )
        packed_index = torch.cat([middle] + packed_groups, dim=0)
        assert _is_permutation(packed_index)
        return packed_index

    def order_from_scores(scores, select_num=None, outlier_mask=None):
        if metric == "rms":
            outlier_scores = scores.get(order_score_key, scores["outlier"]) if isinstance(scores, dict) else scores
            companion_scores = scores.get(companion_score_key, scores["companion"]) if isinstance(scores, dict) else scores
            if select_num is None:
                return plain_order(outlier_scores)
            return packed_order(outlier_scores, companion_scores, select_num, outlier_mask)
        return plain_order(scores)

    def combine_scores(*score_items):
        if isinstance(score_items[0], dict):
            combined = {}
            for key in score_items[0]:
                values = [item[key] for item in score_items if key in item]
                if values and torch.is_tensor(values[0]):
                    stacked = torch.stack([value.float() for value in values], dim=0)
                    combined[key] = stacked.max(dim=0).values.cpu()
                elif values:
                    combined[key] = values[0]
            return combined
        stacked = torch.stack([item.float() for item in score_items], dim=0)
        return stacked.max(dim=0).values.cpu()

    act_orders = {}
    for name, module in model.model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        input_name = name + ".input"
        select_num = select_nums.get(input_name) if select_nums is not None else None
        outlier_mask = None if outlier_masks is None else outlier_masks.get(input_name)
        act_orders[input_name] = order_from_scores(act_scales[input_name], select_num, outlier_mask)

    return act_orders


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError(
            f"Unsupported model type: {config.model_type}. "
            "Supported model families: Llama3 and Qwen3."
        )
    config.use_cache = False
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc


@torch.no_grad()
def get_act_stats(model, dataloader, device_, metric="max", seqlen=2048, reorder_index=None):
    if metric not in SUPPORTED_REORDER_METRICS:
        raise ValueError(f"Unsupported activation metric: {metric}. Supported: {sorted(SUPPORTED_REORDER_METRICS)}")

    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()

        if metric == "rms":
            tensor_float = tensor.float()
            sq_sum = torch.sum(tensor_float * tensor_float, dim=0).cpu()
            max_abs = torch.linalg.norm(tensor_float.abs(), ord=float("inf"), dim=0).cpu()
            current = {
                "outlier_sq_sum": sq_sum,
                "companion_sq_sum": sq_sum.clone(),
                "max_abs": max_abs,
                "count": tensor_float.shape[0],
            }
        else:
            current = torch.linalg.norm(tensor.abs(), ord=float("inf"), dim=0).float().cpu()

        if name not in act_scales:
            act_scales[name] = current
            return

        if metric == "rms":
            act_scales[name]["outlier_sq_sum"] += current["outlier_sq_sum"]
            act_scales[name]["companion_sq_sum"] += current["companion_sq_sum"]
            act_scales[name]["max_abs"] = torch.max(act_scales[name]["max_abs"], current["max_abs"])
            act_scales[name]["count"] += current["count"]
        else:
            act_scales[name] = torch.max(act_scales[name], current)

    def stat_input_hook(module, inputs, output, name):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        y = output[0] if isinstance(output, tuple) else output
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input", x)
        stat_tensor(name + ".output", y)

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        layer_prefix = f"layers.{layer_idx}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}"
                hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=full_name)))

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm") and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            hidden_states = inp[0] if isinstance(inp, tuple) else inp
            inps[cache["i"]] = hidden_states.squeeze(0)
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        input_ids = batch[0] if isinstance(batch, tuple) else batch
        try:
            model(input_ids.to(device))
        except ValueError:
            pass

    assert cache["i"] == nsamples, "Captured samples should be equal to nsamples"

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "norm") and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i].to(device)
        for j in range(nsamples):
            layer_out = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            outs[j] = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        gc.collect()

    for hook in hooks:
        hook.remove()

    if metric == "rms":
        for stats in act_scales.values():
            stats["outlier"] = torch.sqrt(stats["outlier_sq_sum"] / stats["count"])
            stats["companion"] = torch.sqrt(stats["companion_sq_sum"] / stats["count"])
            del stats["outlier_sq_sum"]
            del stats["companion_sq_sum"]
            del stats["count"]

    return act_scales


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps


def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    val_files = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
    traindata = load_dataset(
        "allenai/c4",
        data_files={"validation": val_files},
        split="validation",
        trust_remote_code=True,
    )

    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            encoded = tokenizer(traindata[i]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= seqlen:
                j = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
                inp = encoded.input_ids[:, j : j + seqlen]
                break
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps


def get_pile(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")

    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            encoded = tokenizer(dataset[i]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] >= seqlen:
                j = random.randint(0, encoded.input_ids.shape[1] - seqlen)
                inp = encoded.input_ids[:, j : j + seqlen]
                break
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps


def get_humaneval(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    dataset = load_dataset("openai/openai_humaneval", split="test")
    texts = [item["prompt"] for item in dataset]
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    inps = []
    max_start = max(0, encoded.input_ids.shape[1] - seqlen - 1)
    for _ in range(nsamples):
        i = random.randint(0, max_start) if max_start > 0 else 0
        inp = encoded.input_ids[:, i : i + seqlen]
        if inp.shape[1] < seqlen:
            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), value=tokenizer.eos_token_id)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps


@torch.no_grad()
def search_select_proportions(model, dataloader, device_, seqlen, reorder_index):
    nsamples = len(dataloader)
    device = device_
    select_nums = {}
    average_bits = {}

    print("Preparing inputs for ARCQuant residual channel search...")
    layers = model.model.layers
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    cache = {"inps": None, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            cache["inps"] = inp
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    if isinstance(dataloader, list):
        dataloader = torch.stack(dataloader, dim=0).squeeze(1)

    try:
        model(dataloader.to(device))
    except ValueError:
        pass

    layers[0] = layers[0].module
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    inps = cache["inps"]
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    total_elements = 0
    total_bits = 0

    def stat_input_hook(module, inputs, output, name, act_inputs):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        act_inputs[name + ".input"] = x

    print("Processing layers for ARCQuant residual channel counts...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(device)
        act_inputs = {}
        hooks = []
        layer_prefix = f"layers.{i}"

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}"
                hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=full_name, act_inputs=act_inputs)))

        inps = inps.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if position_embeddings is not None:
            position_embeddings = tuple(t.to(device) for t in position_embeddings)

        with torch.no_grad():
            layer_out = layer(
                inps,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            inps = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        for name, keys in act_inputs.items():
            keys = keys.reshape(-1, keys.shape[-1]).contiguous()
            _, in_features = keys.shape

            if name not in reorder_index:
                print(f"Warning: {name} not found in reorder_index")
                continue

            idx = reorder_index[name].to(device).to(torch.int32)
            keys = keys[:, idx]
            threshold = keys.abs().max(dim=-1, keepdim=True)[0] * 0.125
            select_ratio = (keys.abs() > threshold).sum() / keys.numel()
            select_num = math.ceil(in_features * select_ratio / 64) * 64
            select_num = min(select_num, in_features)

            avg_bits = 4.5 * (in_features + select_num) / in_features
            average_bits[name] = avg_bits
            select_nums[name] = select_num
            total_elements += in_features
            total_bits += 4.5 * (in_features + select_num)

        for hook in hooks:
            hook.remove()

        del act_inputs
        del hooks
        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Average bits is {(total_bits / total_elements):.2f}")
    return select_nums, average_bits


@torch.no_grad()
def build_zero_select_nums(model):
    select_nums = {}
    average_bits = {}
    total_elements = 0
    total_bits = 0

    print("Generating RTN select_num values: all residual channel counts are 0.")
    for i, layer in enumerate(tqdm(model.model.layers)):
        layer_prefix = f"layers.{i}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}.input"
                select_nums[full_name] = 0
                average_bits[full_name] = 4.5
                total_elements += module.in_features
                total_bits += 4.5 * module.in_features
                print(f"{full_name}: 0.00%, avg:4.50")

    print(f"Average bits is {(total_bits / total_elements):.2f}")
    return select_nums, average_bits


@torch.no_grad()
def build_zero_select_nums_silent(model):
    select_nums = {}
    average_bits = {}

    for i, layer in enumerate(model.model.layers):
        layer_prefix = f"layers.{i}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}.input"
                select_nums[full_name] = 0
                average_bits[full_name] = 4.5

    return select_nums, average_bits


@torch.no_grad()
def build_outlier_pack_anchor_counts_auto(
    model,
    act_scales,
    group_size=16,
    metric=None,
    cost_weight=1.0,
    min_improve=0.0,
):
    anchor_counts = {}
    for i, layer in enumerate(tqdm(model.model.layers)):
        layer_prefix = f"layers.{i}"
        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            input_name = f"{layer_prefix}.{name}.input"
            scores = act_scales[input_name]
            if not isinstance(scores, dict) or "outlier" not in scores or "companion" not in scores:
                anchor_counts[input_name] = 0
                continue

            outlier_scores = scores["outlier"].float()
            companion_scores = scores["companion"].float()
            max_group_count = outlier_scores.numel() // group_size if group_size > 0 else 0
            if outlier_scores.numel() == 0 or max_group_count == 0:
                anchor_counts[input_name] = 0
                continue

            outlier_order = torch.argsort(outlier_scores, descending=True).tolist()
            companion_order = torch.argsort(companion_scores, descending=False).tolist()
            chosen_anchor_set = set()
            best_k = 0
            best_delta = float("-inf")
            stop_reason = "no_positive_improvement"

            for k in range(1, max_group_count + 1):
                anchor_idx = outlier_order[k - 1]
                chosen_anchor_set.add(anchor_idx)

                companion_energy = 0.0
                chosen_companions = 0
                for companion_idx in companion_order:
                    if companion_idx in chosen_anchor_set:
                        continue
                    companion_score = float(companion_scores[companion_idx].item())
                    companion_energy += companion_score * companion_score
                    chosen_companions += 1
                    if chosen_companions == group_size - 1:
                        break

                if chosen_companions != group_size - 1:
                    stop_reason = "insufficient_companions"
                    break

                anchor_score = float(outlier_scores[anchor_idx].item())
                delta = anchor_score * anchor_score - cost_weight * companion_energy
                best_delta = max(best_delta, delta)

                if delta > min_improve:
                    best_k = k
                else:
                    stop_reason = "marginal_gain_below_threshold"
                    break

            anchor_counts[input_name] = best_k

    return anchor_counts
