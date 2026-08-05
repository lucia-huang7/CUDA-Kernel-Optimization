# CUDA Extension Development

## Build and validate

Use an existing CUDA-enabled PyTorch installation with matching CUDA toolkit and a C++ compiler. A separate virtual environment can inherit the image's PyTorch without modifying another project:

```bash
mkdir -p results
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
python -m scripts.check_env > results/environment.json
GPU_OPS_BUILD_VERBOSE=1 python -m scripts.smoke_extension
python -m pytest -q
python -m scripts.validate_rmsnorm
```

The first call uses torch.utils.cpp_extension.load to compile the C++ pybind interface and CUDA kernels into a local shared library. Later calls reuse the build cache. Build duration is distinct from kernel latency. Operators allocate outputs on the input device, use the active PyTorch CUDA stream and check launch errors without a device-wide synchronization.

## Numerical contract

RMSNorm is forward-only with contiguous input [rows, width] and contiguous same-device/same-dtype weight [width]. Width is positive and at most 65536. Empty row batches return an empty output. FP32, FP16 and BF16 are supported; gradients and unsupported layouts are rejected. Tests use finite inputs with magnitudes up to 1e4; arbitrary finite magnitudes can still overflow an FP32 sum of squares.

The reference matches Qwen2: accumulate squared inputs in FP32, normalize in FP32, round to the input dtype, then multiply by the learned weight. FP32 reduction order differs across implementations, so tests use fixed numerical tolerances rather than promising bitwise equality. No fast-math option is enabled.

## Source references

- [PyTorch C++ extension loader](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- [Qwen2 RMSNorm in Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py)

## Next stages

Profile normalization, tune access/reduction strategy, add LayerNorm and implement GEMM baselines with shared-memory/register tiling. Kernel speedups and model-level impact require separate measurements.

## Reduction experiment

The baseline entry point remains `gpu_ops.rmsnorm`; `gpu_ops.rmsnorm_warp` selects the optimized reduction. Keep both available for regression and timing comparisons.

```bash
python -m scripts.benchmark_rmsnorm --include-warp --include-compiled --output results/new-comparison.json
python -m scripts.validate_rmsnorm --output results/new-correctness.json
python -m scripts.inspect_cuda_binary
```

Choose new output paths for each run. The committed report generator reads the named historical comparison files, preserving the original dataset. Static disassembly/resource records require `cuobjdump` and the built shared library.

The timing runner captures separate CUDA Graph sequences for each method, randomizes method order across 30 repetitions and also saves ordinary Python-call wall timing. Profiling is separate. The recorded NCU attempt failed with `ERR_NVGPUCTRPERM`; no hardware-counter utilization claim is made. No host-driver permissions were changed.

The plotting script uses saved timing data. Install `requirements-figures.txt` in a plotting environment and run `python scripts/plot_rmsnorm.py`. The figure uses DejaVu Sans plot text, Times New Roman captions and outlined SVG glyphs.
