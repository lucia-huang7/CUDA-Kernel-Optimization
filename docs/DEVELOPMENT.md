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

## Fixed-width layout experiment

`rmsnorm_vector128` is a width-1536 specialization with four-element packs and 128 threads per row. BF16/FP16 packs require 8-byte alignment, FP32 packs require 16-byte alignment. Misaligned contiguous storage uses `rmsnorm_cached`; other widths use the general `rmsnorm_warp` implementation. All variants remain separately callable for ablation.

```bash
python -m scripts.benchmark_rmsnorm --include-warp --include-specialized --include-compiled --methods cuda_warp cuda_vector128 pytorch_compiled --output results/new-specialized-run.json
python -m scripts.validate_rmsnorm --output results/new-specialized-correctness.json
python -m scripts.inspect_specialized_binary
```

Select fresh result paths. `report_specialized_rmsnorm.py` and `plot_specialized_rmsnorm.py` use the committed historical filenames. The first tuning snapshot and later control/confirmation snapshots correspond to different source revisions, recorded by their hashes.

Generated Inductor inspection is read-only. Static binary evidence confirms wide loads/stores and no local-memory instruction sites in the selected compiled kernels; it does not establish achieved memory throughput.

The recorded CUDA 12.8 environment reports a `cuKernelGetFunction` invalid-handle diagnostic when the extension is loaded after a PyTorch CUDA kernel has run. Load native kernels during application startup, before CUDA tensor operations:

```python
from gpu_ops import initialize
initialize()
```

The sanitizer workload runner uses this order without launching a warm-up kernel or changing API error reporting:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python -m scripts.sanitizer_workload pytest
compute-sanitizer --tool memcheck --error-exitcode 99 python -m scripts.sanitizer_workload alignment
compute-sanitizer --tool memcheck --error-exitcode 99 python -m scripts.sanitizer_workload numerical --output results/new-sanitizer-correctness.json
compute-sanitizer --tool synccheck --error-exitcode 99 python -m scripts.sanitizer_workload alignment
```

[Initialization-order diagnosis and L4 verification](../results/sanitizer-diagnosis.md) includes the failing controls, a clean-build comparison and a deliberate out-of-bounds negative control. Historical failure logs remain unchanged. The startup-loading workaround does not claim a vendor-internal driver/runtime fix.

Reference: [CUDA 12.8 size and alignment requirements](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#device-memory-accesses).


## Affine LayerNorm

`gpu_ops.layernorm` retains the shared-tree baseline. `layernorm_warp` uses a general 128-thread centered two-pass reduction. `layernorm_cached` and `layernorm_vector` specialize width 1536, retaining 12 FP32 values per thread. The vector entry validates all three input-pointer alignments and falls back to scalar caching when needed; other widths use the general warp kernel. Statistics and affine arithmetic use FP32, with only the final output cast to the input dtype.

```bash
python -m scripts.validate_layernorm --output results/new-layernorm-correctness.json
python -m scripts.benchmark_layernorm --compiled --methods cuda_shared cuda_vector pytorch_native pytorch_compiled --output results/new-layernorm-comparison.json
python -m scripts.profile_layernorm --prefix results/new-layernorm-profile
compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 99 python -m scripts.sanitizer_workload pytest
compute-sanitizer --tool racecheck --error-exitcode 99 python -c "from gpu_ops import initialize; initialize(); import pytest; raise SystemExit(pytest.main(['-q','tests/test_layernorm.py']))"
```

Use fresh paths. Keep at most six static row shapes per compiled-reference process; run other widths/dtypes in separate processes. Graph replay excludes Python dispatch; the same runner also records ordinary call wall times. The FP64 oracle executes on the CPU, separating the expected calculation from the CUDA kernels under test. The original GPU FP64 mean Racecheck report and PyTorch-only reproducer remain in the evidence.

[LayerNorm experiment and results](../results/layernorm.md). Plotting follows the established DejaVu Sans axes/legend and Times New Roman caption style.
