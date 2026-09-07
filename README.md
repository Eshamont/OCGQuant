# OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization

[![arXiv](https://img.shields.io/badge/arXiv-2609.00066-b31b1b.svg)](https://arxiv.org/abs/2609.00066)

This repository contains the PyTorch implementation of the EMNLP 2026 paper [OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization](https://arxiv.org/abs/2609.00066).

## Installation

```bash
git clone --recurse-submodules https://github.com/Eshamont/OCGQuant.git
cd OCGQuant
```


Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is available in your environment.

```bash
conda create -n ocgquant python=3.10 -y
conda activate ocgquant

sudo apt-get update
sudo apt-get install build-essential cmake python3-dev

conda install pybind11
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```




## Usage

> **Note:**
> - The CUDA kernels are currently configured for NVIDIA Blackwell SM120 GPUs.
> - Non-Blackwell GPUs only support PyTorch fake-quantization experiments, but the real NVFP4 CUDA kernels and corresponding hardware acceleration are unavailable.


### Building Kernels

```bash
cd kernels
bash remake.sh
cd ..
```


### Preprocessing

Before evaluation, OCGQuant requires precomputed `.pt` files.

```bash
python reorder_indices.py --model /path/to/model --samples 128 --seqlen 2048 --dataset wikitext2 --method ocgquant
```

The generated files will be saved to `./saved/`.

Run preprocessing before PPL or accuracy evaluation so the required files exist in `./saved/`.


### PPL

```bash
python main.py /path/to/model --dataset wikitext2 --method ocgquant --quant_type NVFP4 --eval_ppl
```


### Accuracy Evaluation

The example below evaluates common zero-shot tasks.

```bash
python main.py /path/to/model --dataset wikitext2 --method ocgquant --quant_type NVFP4 --tasks piqa,arc_challenge,boolq,hellaswag,lambada_openai,arc_easy --lm_eval_num_fewshot 0 --lm_eval_limit -1
```

## Acknowledgements

This project builds on several excellent open-source efforts. We sincerely thank the community for their contributions:
- [ARCQuant](https://github.com/actypedef/ARCQuant)
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{yao2026ocgquant,
  title={OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization},
  author={Yao, Yishan and Li, Binjun and Yi, Hanling and Li, Pengyu and Liu, Xiaoqing and Yang, Zihan and Yu, Xiaotian and Yu, Zhiwen},
  journal={arXiv preprint arXiv:2609.00066},
  year={2026}
}
```
