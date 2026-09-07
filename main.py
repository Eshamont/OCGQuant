import torch
from collections import defaultdict
import argparse
import inspect
import os
from pathlib import Path

from model.model_utils import reorder_model_llama, reorder_model_qwen3
from model.parallel_utils import map_layers_to_multi_gpus
from model.datautils import DEV, get_loaders
from model.eval import eval_ppl
from model.quantize import use_nvfp4_kernel_path

import time


SAVED_DIR = Path(__file__).resolve().parent / "saved"



def build_rtn_metadata(model):
    reorder_index = {}
    select_nums = {}

    for i, layer in enumerate(model.model.layers):
        layer_prefix = f"layers.{i}"
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                full_name = f"{layer_prefix}.{name}.input"
                reorder_index[full_name] = torch.arange(module.in_features, dtype=torch.int16)
                select_nums[full_name] = 0

    return reorder_index, select_nums


def cache_metric_name(metric):
    return metric



METHOD_METRICS = {
    "arcquant": "max",
    "ocgquant": "rms",
}


def cache_method_name(method):
    return method


def force_rtn_select_nums(select_nums):
    return {name: 0 for name in select_nums}


def resolve_cache_file(primary, legacy=None):
    primary = Path(primary)
    if primary.is_file():
        return primary
    if legacy is None:
        return primary
    legacy_files = legacy if isinstance(legacy, (list, tuple)) else [legacy]
    for legacy_file in legacy_files:
        legacy_file = Path(legacy_file)
        if legacy_file.is_file():
            print(f"Using legacy cache file: {legacy_file}")
            return legacy_file
    return primary



def calibrate_activation_global_scales(model, args):
    from model.qLinearLayer import get_act_global_scale_summary, set_act_global_scale_tracking

    print(
        f"Calibrating NVFP4 activation global scales: dataset={args.dataset}, "
        f"samples={args.samples}, seqlen={args.seqlen}, seed={args.seed}"
    )
    model = model.to(DEV)
    model.eval()
    fusion_states = []
    for module in model.modules():
        if hasattr(module, "fuse_rmsnorm_reorder"):
            fusion_states.append((module, module.fuse_rmsnorm_reorder))
            module.fuse_rmsnorm_reorder = False

    try:
        set_act_global_scale_tracking(model, enabled=True, reset=True)

        trainloader, _ = get_loaders(
            args.dataset,
            nsamples=args.samples,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.model,
        )
        calibration_model = getattr(model, "model", model)
        with torch.no_grad():
            for sample in trainloader:
                input_ids = sample[0] if isinstance(sample, tuple) else sample
                calibration_model(input_ids.to(DEV))
    finally:
        set_act_global_scale_tracking(model, enabled=False)
        for module, enabled in fusion_states:
            module.fuse_rmsnorm_reorder = enabled

    summary = get_act_global_scale_summary(model)
    if summary is not None:
        print(
            "Activation global scales calibrated: "
            f"{summary['calibrated']}/{summary['count']} layers, "
            f"min={summary['min']:.6g}, max={summary['max']:.6g}"
        )
    return model


def patch_transformers_lm_eval_compat():
    import transformers

    if not hasattr(transformers, "HybridCache"):
        class HybridCache(transformers.DynamicCache):
            def __init__(self, *args, **kwargs):
                try:
                    super().__init__(*args, **kwargs)
                except TypeError:
                    super().__init__()

        transformers.HybridCache = HybridCache

    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(transformers, "AutoModelForImageTextToText"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText



def load_quant_metadata(args, model, index_filename, select_num_filename):
    if args.method == "rtn":
        print("Method: rtn. Building identity reorder metadata and forcing select_num=0.")
        return build_rtn_metadata(model)

    if not os.path.isfile(index_filename):
        raise FileNotFoundError(
            f"{args.method} method requires cached reorder index: {index_filename}. "
            "Run reorder_indices.py first."
        )

    print("Loading cached reordering index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)

    if args.method == "ocgquant":
        # print("Method: ocgquant. Using cached reorder_index and forcing select_num=0.")
        return reorder_index, force_rtn_select_nums(reorder_index)

    if args.method == "arcquant":
        if not os.path.isfile(select_num_filename):
            raise FileNotFoundError(
                f"arcquant method requires cached select_num: {select_num_filename}. "
                "Run reorder_indices.py first."
            )
        print("Method: arcquant. Using cached reorder_index and select_num residual channels.")
        select_nums = torch.load(select_num_filename, weights_only=False)
        return reorder_index, select_nums

    raise ValueError(f"Unsupported method: {args.method}")


def apply_ocg_gptq(model, args, reorder_index):
    if args.method != "ocgquant":
        return model

    from model.gptq import ocg_gptq_quantization, build_calibration_data

    print(
        f"Running OCGQuant for final reorder quantization: dataset={args.dataset}, "
        f"samples={args.samples}, seqlen={args.seqlen}, seed={args.seed}"
    )
    calibration_data = build_calibration_data(
        args.model, args.dataset, args.seed, args.seqlen, args.samples, get_loaders
    )
    return ocg_gptq_quantization(model, calibration_data, reorder_index, args.quant_type, DEV, args)


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    # model.seqlen = 2048
    return model

def get_qwen3(model):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model)
    if config.model_type != "qwen3":
        raise ValueError(f"Expected a Qwen3 checkpoint, got model type: {config.model_type}.")
    return AutoModelForCausalLM.from_pretrained(model, config=config, torch_dtype="auto")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Path to a Llama3 or Qwen3 checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='rtn',
        choices=['rtn', 'arcquant', 'ocgquant'],
        help='Quantization method: rtn, arcquant, or ocgquant.'
    )
   
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )

    parser.add_argument(
        '--tasks', type=str, default=None,
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )

    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        "--lm_eval_apply_chat_template", action="store_true",
        help="Apply the tokenizer chat template in lm-eval. Useful for instruct CoT tasks.",
    )
    parser.add_argument(
        "--lm_eval_fewshot_as_multiturn", action="store_true",
        help="Format few-shot examples as multiturn conversations in lm-eval.",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"], 
        help="The calibration dataset to use."
    )
    parser.add_argument(
        "--quant_type", type=str, default="NVFP4",
        help="Data type for W and A quantization."
    )
    parser.add_argument("--samples", type=int, default=128, help="Number of calibration samples for GPTQ.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Calibration sequence length for GPTQ.")
    parser.add_argument("--quantization_order", type=str, default="default", choices=["default", "activation"])
    parser.add_argument("--rel_damp", type=float, default=0.1)
    parser.add_argument("--cpu_offload_modules", action="store_true")
    parser.add_argument("--cpu_offload_activations", action="store_true")
    parser.add_argument(
        "--fuse_rmsnorm_reorder",
        action="store_true",
        help="Strict semantic RMSNorm + reorder quantization fusion for q_proj/up_proj inputs.",
    )
    parser.add_argument(
        "--fuse_rmsnorm_reorder_kernel",
        action="store_true",
        help=(
            "Use the true CUDA RMSNorm+reorder+NVFP4 activation quantization kernel when supported. "
            "Falls back to the strict semantic path if a frozen activation scale is unavailable."
        ),
    )
  
    
    args = parser.parse_args()
    if args.quant_type != "NVFP4":
        parser.error("--quant_type only supports NVFP4 now.")
    if args.method == "ocgquant" and not args.fuse_rmsnorm_reorder_kernel:
        # print("Enabling --fuse_rmsnorm_reorder_kernel by default for ocgquant.")
        args.fuse_rmsnorm_reorder_kernel = True
    uses_kernel_path = use_nvfp4_kernel_path()
    path_name = "CUDA kernel" if uses_kernel_path else "PyTorch fake"
    print(f"NVFP4 group size: 16 ({path_name} path)")
    if not uses_kernel_path and args.fuse_rmsnorm_reorder_kernel:
        print("Disabling --fuse_rmsnorm_reorder_kernel because the NVFP4 CUDA kernel path requires a built ocgcuda extension and an NVIDIA Blackwell SM120 GPU.")
        args.fuse_rmsnorm_reorder_kernel = False
    if args.method == "arcquant":
        os.environ["ARC_DISABLE_REORDER_META"] = "1"

    model_name = args.model.split('/')[-2] if len(args.model.split('/')[-1]) == 0 else args.model.split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        reorder_model_func = reorder_model_llama
       
    elif "qwen" in args.model.lower():
        model = get_qwen3(args.model)
        reorder_model_func = reorder_model_qwen3
    
    else:
        raise ValueError("Supported model families: Llama3 and Qwen3.")
       
    model.eval()

    if args.method == "rtn":
        reorder_index, select_nums = load_quant_metadata(args, model, None, None)
    else:
        dataset_name = args.dataset.lower()
        metric = METHOD_METRICS[args.method]
        output_method_name = cache_method_name(args.method)
        legacy_metric_names = [cache_metric_name(metric)]
        index_filename = resolve_cache_file(
            SAVED_DIR / f"{model_name.lower()}_reorder_index_{dataset_name}_{output_method_name}.pt",
            [
                SAVED_DIR / f"{model_name.lower()}_reorder_index_{dataset_name}_{name}.pt"
                for name in legacy_metric_names
            ],
        )
        select_num_filename = resolve_cache_file(
            SAVED_DIR / f"{model_name.lower()}_select_num_{dataset_name}_{output_method_name}.pt",
            [
                SAVED_DIR / f"{model_name.lower()}_select_num_{dataset_name}_{name}.pt"
                for name in legacy_metric_names
            ],
        )
        act_scales_filename = resolve_cache_file(
            SAVED_DIR / f"{model_name.lower()}_act_scales_{dataset_name}_{output_method_name}.pt",
            [
                SAVED_DIR / f"{model_name.lower()}_act_scales_{dataset_name}_{name}.pt"
                for name in legacy_metric_names
            ],
        )


        reorder_index, select_nums = load_quant_metadata(args, model, index_filename, select_num_filename)

        if os.path.isfile(act_scales_filename):
            act_scales = torch.load(act_scales_filename, weights_only=False)

    model = apply_ocg_gptq(model, args, reorder_index)
    
    torch.cuda.reset_max_memory_allocated()
    print(f"Applying quantized model wrappers for {args.method} method...")
    start_time=time.time()
    model = reorder_model_func(
        model,
        device=DEV,
        kv_cache=args.kv_cache,
        reorder_index=reorder_index,
        select_nums=select_nums,
        quant_type=args.quant_type,
        fuse_rmsnorm_reorder=args.fuse_rmsnorm_reorder,
        fuse_rmsnorm_reorder_kernel=args.fuse_rmsnorm_reorder_kernel,
        r5_reorder_absorb=(args.method == "ocgquant"),
    )
    end_time=time.time()
    peak_memory = torch.cuda.max_memory_allocated()


    # print(model)
    print(f"Quantized Model Size: {peak_memory/(1024*1024*1024):.2f} GB")
    print(f"Quantized Type is: {args.quant_type} ")
    if args.method == "arcquant":
        print("ARCQuant method: cached residual channel counts are enabled.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    if args.method in {"arcquant", "ocgquant"}:
        model = calibrate_activation_global_scales(model, args)

    bsz = "auto"
    if args.tasks is not None:
        if 'mmlu' in args.tasks :
            bsz = 2
        elif "gsm8k" in args.tasks:
            bsz = 2
 
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # map_layers_to_multi_gpus(lm.model.model.layers)
    # input_device = lm.model.model.layers[0].device
    # output_device = lm.model.model.layers[-1].device
    # assert input_device == output_device
    # lm._device = input_device
    # lm.model.model.embed_tokens.to(input_device)
    # lm.model.model.norm.to(output_device)
    # lm.model.lm_head.to(output_device)
    if args.tasks is not None:
        model = model.to(DEV)

        
    if args.eval_ppl:
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=2048
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_ppl(model, testloader, 'cuda')

            print(f"Result,{dataset},{ppl:.3f}")

    
            
    if args.tasks is not None:
        patch_transformers_lm_eval_compat()

        from lm_eval import evaluator as lm_evaluator
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(model, batch_size=bsz)
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

        lm._device = DEV
        lm._model = lm._model.to(lm._device)

        task_manager = TaskManager(include_path="./lm_eval_tasks" if os.path.isdir("./lm_eval_tasks") else None)
        task_names = args.tasks.split(',')

        eval_kwargs = {
            "tasks": task_names,
            "num_fewshot": args.lm_eval_num_fewshot,
            "limit": None if args.lm_eval_limit == -1 else args.lm_eval_limit,
            "batch_size": bsz,
            "task_manager": task_manager,
        }
        simple_eval_params = inspect.signature(lm_evaluator.simple_evaluate).parameters
        optional_eval_kwargs = {
            "apply_chat_template": args.lm_eval_apply_chat_template,
            "fewshot_as_multiturn": args.lm_eval_fewshot_as_multiturn,
        }
        for key, value in optional_eval_kwargs.items():
            if key in simple_eval_params:
                eval_kwargs[key] = value
            elif value:
                print(f"Warning: installed lm-eval does not support {key}; ignoring this option.")

        results = lm_evaluator.simple_evaluate(lm, **eval_kwargs)

        table_results = make_table(results)
        print(table_results)
        import logging
        from datetime import datetime

        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        log_filename = f"./results/log_{model_name.lower()}_{args.tasks}_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
                            filename=log_filename,
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                        )
        logging.info(f"Results for {model_name.lower()} with {args.method} on {args.tasks}:\n{table_results}")
  
