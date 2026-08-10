# GPU Kernel Lab

CUDA C++ operator development with PyTorch bindings, numerical tests and hardware profiling. The project covers normalization and matrix multiplication, with implementations and measurements added in stages.

## Validation

**NVIDIA L4 · PyTorch 2.8.0 · CUDA 12.8 · FP32 / FP16 / BF16**

| Check | Result |
|---|---:|
| Initial extension build and first load | 17.49 s |
| Python-to-CUDA smoke cases | 18 passed |
| RMSNorm correctness and execution semantics, seven variants | 480 passed |
| Separate RMSNorm numerical cases, seven variants | 924 passed |
| Full test suite, RMSNorm / LayerNorm / GEMM | 1330 passed |
| LayerNorm numerical cases, four variants | 720 passed |
| GEMM numerical cases, all candidates and controls | 768 passed |
| GEMM synchronization / race checks | 192 cases each; 0 errors / hazards |
| Compute Sanitizer memcheck with startup loading | 1330 tests; 0 errors |
| RMSNorm alignment and row tails: memcheck / synccheck / racecheck | 60 cases each; 0 errors / hazards |

[Validation results](results/validation.md) · [Current numerical evidence](results/rmsnorm-specialized-final-correctness.json) · [Sanitizer diagnosis](results/sanitizer-diagnosis.md)

## FP32 GEMM optimization

Shared-memory tiling, register blocking and vectorized tile access reduce repeated loads and retain multiple output accumulators per thread. A guarded split-K path increases parallelism for small-M, long-K matrices. Independent L4 confirmation shows **2.34×** and **3.79×** speedups over the CUDA baseline on the two rectangular target shapes, alongside PyTorch and direct cuBLAS comparisons with TF32 disabled.

![GEMM performance](assets/figures/gemm_v1.png)

| M × N × K | CUDA baseline | Shared tile | Tuned CUDA | PyTorch | cuBLAS |
|---|---:|---:|---:|---:|---:|
| 16 × 4096 × 4096 | 617.044 μs | 464.093 μs | 263.518 μs | 281.820 μs | 282.527 μs |
| 4096 × 4096 × 32 | 1004.693 μs | 791.953 μs | 265.044 μs | 274.468 μs | 271.799 μs |
| 512 × 512 × 512 | 191.522 μs | 149.533 μs | 68.668 μs | 32.454 μs | 32.423 μs |

FP32, median of 30 interleaved repetitions with 20 calls per replay. All candidate configurations, five additional shapes, ordinary-call timings and resource measurements are in the [full experiment](results/gemm.md).

## LayerNorm optimization

Affine LayerNorm combines stable centered statistics, warp-shuffle reductions, register reuse and four-element vector loads/stores. On L4, the width-1536 vector kernel reduces static block barrier sites from **20 to 4** and reaches **9.795 μs** at 2048 rows, compared with **38.930 μs** for the CUDA baseline and **19.415 μs** for native PyTorch.

![LayerNorm performance](assets/figures/layernorm_v1.png)

| Rows | CUDA baseline | Vectorized CUDA | PyTorch native | PyTorch compiled |
|---:|---:|---:|---:|---:|
| 1 | 3.587 μs | 1.980 μs | 3.138 μs | 1.775 μs |
| 8 | 3.684 μs | 2.024 μs | 3.196 μs | 1.816 μs |
| 32 | 3.694 μs | 2.117 μs | 3.252 μs | 1.935 μs |
| 128 | 4.361 μs | 2.425 μs | 3.645 μs | 2.676 μs |
| 512 | 9.741 μs | 3.675 μs | 5.813 μs | 7.178 μs |
| 2048 | 38.930 μs | 9.795 μs | 19.415 μs | 29.039 μs |

BF16, width 1536; median of 30 interleaved repeats with 100 calls per replay. Independent confirmation, five additional row counts, three dtypes and fallback-width measurements are included in the [full experiment](results/layernorm.md).

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
| GEMM kernels and direct cuBLAS control | [csrc/gemm.cu](csrc/gemm.cu) |
| GEMM reference and precision controls | [reference/gemm.py](reference/gemm.py) |
| LayerNorm kernels | [csrc/layernorm.cu](csrc/layernorm.cu) |
| LayerNorm reference and native control | [reference/layernorm.py](reference/layernorm.py) |
| RMSNorm reference | [reference/rmsnorm.py](reference/rmsnorm.py) |

All RMSNorm variants use FP32 reduction and Qwen2-compatible cast order. The width-1536 vector path uses 128 threads per row; other widths keep the general warp fallback. It supports contiguous two-dimensional FP32, FP16 and BF16 tensors. [Development workflow](docs/DEVELOPMENT.md).
