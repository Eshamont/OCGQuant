from datasets import load_dataset
import torch.nn as nn
import gc
from utilize import * 
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
import tqdm
import argparse
import math
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
SAVED_DIR = REPOSITORY_ROOT / "saved"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or Hugging Face model identifier.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "humaneval", "pile"],
        help="Calibration dataset.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ocgquant",
        choices=["rtn", "arcquant", "ocgquant"],
        help="Quantization method. RTN does not require cached reorder metadata.",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    return parser.parse_args()

METHOD_METRICS = {
    "arcquant": "max",
    "ocgquant": "rms",
}


DATASET_LOADERS = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
    "pile": get_pile,
    "humaneval": get_humaneval
}


def cache_method_name(method):
    return method


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


def main():
    args = parse_args()
    if args.method == "rtn":
        print("RTN does not require precomputed reorder metadata. Run main.py --method rtn directly.")
        return

    metric = METHOD_METRICS[args.method]
    model, enc = load_model(args.model)
    model_name = args.model.rstrip("/").split("/")[-1]
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    start_time = time.time()
    
    print(f"Using {args.dataset} dataset for calibration.")
    get_dataset = DATASET_LOADERS[args.dataset]

    dataset_name = args.dataset.lower()
    output_method_name = cache_method_name(args.method)
    legacy_metric_names = [metric]
    act_scales_filename = resolve_cache_file(
        SAVED_DIR / f"{model_name.lower()}_act_scales_{dataset_name}_{output_method_name}.pt",
        [
            SAVED_DIR / f"{model_name.lower()}_act_scales_{dataset_name}_{name}.pt"
            for name in legacy_metric_names
        ],
    )
    anchor_count_filename = SAVED_DIR / f"{model_name.lower()}_anchor_count_{dataset_name}_{output_method_name}.pt"

    print("Getting activation stats...")
    if not act_scales_filename.is_file():
        print("Generating activation stats...")
        dataloader, _ = get_dataset(
            nsamples=args.samples, seed=0, seqlen=args.seqlen, tokenizer=enc
        )

        act_scales = get_act_stats(
            model, dataloader, "cuda:0", metric=metric, seqlen=args.seqlen
        )
        torch.save(act_scales, act_scales_filename)
        del dataloader
    else:
        print("Loading pre-saved activation stats...")
        act_scales = torch.load(act_scales_filename)


    anchor_counts = None
    if args.method == "ocgquant":
        print("Building max-zero packed reorder index with automatic anchor-count search and RMS companions...")
        anchor_counts = build_outlier_pack_anchor_counts_auto(
            model,
            act_scales,
            metric=metric,
        )
        pack_mode = "compact"
        outlier_masks = None
        order_score_key = "outlier"
        companion_score_key = "companion"
        # print(f"Using outlier-pack reorder pack_mode={pack_mode}, order_score_key={order_score_key}")
        reorder_index = get_reorder_index(
            model,
            act_scales,
            metric=metric,
            select_nums=anchor_counts,
            pack_mode=pack_mode,
            outlier_masks=outlier_masks,
            order_score_key=order_score_key,
            companion_score_key=companion_score_key,
        )
        select_num, average_bits = build_zero_select_nums_silent(model)
    elif args.method == "arcquant":
        print("Getting reorder index for ARCQuant max...")
        reorder_index = get_reorder_index(
            model,
            act_scales,
            metric=metric,
        )
        
        print("Getting proportions...")

        _, inps = get_dataset(
                    nsamples=32, seed=0, tokenizer=enc, seqlen=args.seqlen
                )
        select_num, average_bits = search_select_proportions(model, inps, "cuda", args.seqlen, reorder_index)
    else:
        raise ValueError(f"Unsupported preprocessing method: {args.method}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    reorder_filename = SAVED_DIR / f"{model_name.lower()}_reorder_index_{dataset_name}_{output_method_name}.pt"
    select_num_filename = SAVED_DIR / f"{model_name.lower()}_select_num_{dataset_name}_{output_method_name}.pt"
    avg_bits_filename = SAVED_DIR / f"{model_name.lower()}_average_bits_{dataset_name}_{output_method_name}.pt"

    print(f"Saving reorder index to {reorder_filename}")
    torch.save(reorder_index, reorder_filename)
    print(f"Saving select num to {select_num_filename}")
    torch.save(select_num, select_num_filename)
    print(f"Saving average bits to {avg_bits_filename}")
    torch.save(average_bits, avg_bits_filename)
    if anchor_counts is not None:
        print(f"Saving anchor counts to {anchor_count_filename}")
        torch.save(anchor_counts, anchor_count_filename)
    
if __name__ == "__main__":
    main()
