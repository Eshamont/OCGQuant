# CUDA kernels

This directory contains the custom NVFP4 quantization, reorder, RMSNorm, and
matrix-multiplication kernels used by OCGQuant. The current CMake configuration
targets NVIDIA Blackwell SM120 GPUs.

## Layout

- `src/`: CUDA implementations and Python bindings.
- `include/`: project and FlashInfer headers.
- `benchmark/`: native CUDA benchmarks.
- `main.py`: CUDA extension smoke test.
- `bench.py`: Python benchmark driver.
- `remake.sh`: clean configure and build.
- `makerun.sh`: incremental build followed by the smoke test.

The build output is written to `kernels/build/` and is intentionally ignored by
Git.

## Build

From the repository root, run:

```bash
bash kernels/remake.sh
```
