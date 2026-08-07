# GPU Kernel Lab

CUDA C++ operator development with PyTorch bindings, numerical tests and hardware profiling. The project covers normalization and matrix multiplication, with implementations and measurements added in stages.

## Validation

**NVIDIA L4 · PyTorch 2.8.0 · CUDA 12.8 · FP32 / FP16 / BF16**

| Check | Result |
|---|---:|
| CUDA extension compile and first load | 17.49 s |
| Python-to-CUDA smoke cases | 18 passed |
| Correctness and execution-semantics tests, seven variants | 480 passed |
| Separate RMSNorm numerical cases, seven variants | 924 passed |
| Compute Sanitizer memcheck with startup loading | 480 tests; 0 errors |
| Alignment and row tails: memcheck / synccheck / racecheck | 60 cases each; 0 errors / hazards |

[Validation results](results/validation.md) · [Current numerical evidence](results/rmsnorm-specialized-final-correctness.json) · [Sanitizer diagnosis](results/sanitizer-diagnosis.md)

## Width-1536 RMSNorm specialization

Register reuse, fixed-width unrolling and a 128-thread vectorized row layout reduce GPU replay latency by **37.4%–58.2%** on the six confirmation shapes versus the previous warp kernel. At 2048 rows, the new kernel reaches **9.359 μs**, compared with **17.905 μs** for PyTorch compiled.

![Specialized RMSNorm performance](assets/figures/rmsnorm_specialized_v1.png)

| Rows | Previous CUDA | Vector 128 | PyTorch compiled |
|---:|---:|---:|---:|
| 1 | 3.009 μs | 1.674 μs | 1.426 μs |
| 128 | 3.309 μs | 2.070 μs | 1.897 μs |
| 512 | 6.303 μs | 3.044 μs | 4.293 μs |
| 2048 | 22.379 μs | 9.359 μs | 17.905 μs |

BF16, width 1536; median of 30 interleaved repeats with 100 calls per replay. A fresh-process confirmation and five additional row counts accompany the candidate ablations. [Full experiment](results/rmsnorm-specialized.md).

## RMSNorm reduction optimization

Warp-shuffle reduction cuts block-wide barriers from **9 to 2** and static shared storage from **1024 to 32 bytes per block**. On L4, BF16 GPU replay latency falls by **5.0%–14.3%** across 1–2048 rows at width 1536.

![RMSNorm reduction performance](assets/figures/rmsnorm_reduction_v1.png)

| Rows | CUDA baseline | Warp reduction | Latency reduction |
|---:|---:|---:|---:|
| 1 | 3.183 us | 3.012 us | 5.3% |
| 128 | 3.616 us | 3.288 us | 9.1% |
| 512 | 7.528 us | 6.587 us | 12.5% |
| 2048 | 29.852 us | 25.579 us | 14.3% |

Median of 30 interleaved repeats, each replaying 100 calls. The chart includes the faster PyTorch compiled control. [Full experiment and raw evidence](results/rmsnorm-optimization.md).

## Implementation

| Component | Source |
|---|---|
| Python entry points | [gpu_ops/ops.py](gpu_ops/ops.py) |
| Extension loader | [gpu_ops/extension.py](gpu_ops/extension.py) |
| Pybind interface | [csrc/bindings.cpp](csrc/bindings.cpp) |
| CUDA kernels and launch wrappers | [csrc/kernels.cu](csrc/kernels.cu) |
| Fixed-width caching and vector layouts | [csrc/rmsnorm_specialized.cu](csrc/rmsnorm_specialized.cu) |
| RMSNorm reference | [reference/rmsnorm.py](reference/rmsnorm.py) |

All RMSNorm variants use FP32 reduction and Qwen2-compatible cast order. The width-1536 vector path uses 128 threads per row; other widths keep the general warp fallback. It supports contiguous two-dimensional FP32, FP16 and BF16 tensors. [Development workflow](docs/DEVELOPMENT.md).
