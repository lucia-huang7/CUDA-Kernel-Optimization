# LayerNorm: CUDA implementation and L4 optimization

The vectorized width-1536 kernel reduces GPU replay latency by **42.7%–74.8%** versus the shared-tree CUDA baseline on six independent confirmation shapes. It is **1.50×–1.98×** faster than native PyTorch on these BF16 shapes. At 2048 rows it takes **9.795 μs** versus **19.415 μs** for native PyTorch and **29.039 μs** for compiled reference.

## Semantics and implementation

`y = ((x - mean(x)) / sqrt(mean((x - mean(x))²) + eps)) * weight + bias`, with population variance, FP32 statistics/affine arithmetic and a final cast to the input dtype. Default epsilon is 1e-5. Unlike the RMSNorm path, normalization is not rounded to BF16/FP16 before affine scaling.

The forward-only interface accepts contiguous CUDA tensors of shape [rows, width], width 1–65536, with contiguous same-device and same-dtype weight/bias vectors. FP32, FP16 and BF16 are supported. Empty rows return an empty output. Autograd inputs and invalid contracts are rejected.

Statistics are centered around the first row element: reduce x-anchor, then reduce squared deviations in that shifted coordinate system. This avoids subtracting two large raw moments and preserves small variations on an offset input. An independent CPU FP64 oracle checks the mathematical result.

## Locate, change, measure

The initial implementation passed 560 combined tests. Its first BF16 run measured 47.695 μs at 2048 rows, versus native PyTorch 23.539 μs. The profiler observed 8 CUDA launches per decomposed reference call, 1 for native PyTorch and 1 for the custom kernel. Native PyTorch was already fused.

| Variant | Mechanism | Width-1536 threads/row | Static block barrier sites |
|---|---|---:|---:|
| Shared baseline | Shared-tree mean and variance; input reread | 256 | 20 |
| Warp | Warp shuffles and two separate shared scratch arrays | 128 | 4 |
| Cached | Fixed-width unrolling; retain 12 FP32 values/thread | 128 | 4 |
| Vector | Cached values plus four-element input/affine/output packs | 128 | 4 |

All four CUDA variants retain the same anchored centered-variance formula. Separate mean/variance scratch arrays prevent reuse before all warps finish reading. Vector packs require 8-byte alignment for FP16/BF16 and 16-byte alignment for FP32; any misaligned input, weight or bias selects the scalar cached path. Other widths use the general warp path.

### Candidate ablation

GPU replay medians in μs, from one interleaved tuning run. Failed or weaker candidates remain in the table.

| Rows | Shared | Warp | Cached | Vector | Native | Compiled |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.574 | 3.941 | 3.283 | 1.971 | 3.130 | 1.766 |
| 8 | 3.677 | 4.040 | 3.366 | 2.018 | 3.194 | 1.807 |
| 32 | 3.696 | 4.067 | 3.392 | 2.121 | 3.250 | 1.937 |
| 128 | 4.374 | 4.346 | 3.621 | 2.447 | 3.659 | 2.684 |
| 512 | 10.141 | 6.160 | 5.050 | 3.721 | 5.939 | 7.474 |
| 2048 | 47.722 | 25.033 | 19.208 | 11.809 | 23.234 | 35.347 |

Warp reduction alone regressed on small inputs. Register retention helped, and vector packs produced the strongest overall candidate. These are controlled empirical comparisons, not hardware stall-counter measurements.

## Independent confirmation

NVIDIA L4, PyTorch 2.8.0+cu128, CUDA 12.8, BF16, width 1536. Each median uses 30 randomized interleaved repetitions, 100 calls per CUDA Graph replay, 20 warm-up calls and 5 warm-up replays. First-call compilation/loading is saved separately. Fixed buffers are reused; these are warm-cache microbenchmarks.

![LayerNorm latency](../assets/figures/layernorm_v1.png)

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1536 | bfloat16 | 3.587 | 1.980 | 3.138 | 1.775 |
| 8 | 1536 | bfloat16 | 3.684 | 2.024 | 3.196 | 1.816 |
| 32 | 1536 | bfloat16 | 3.694 | 2.117 | 3.252 | 1.935 |
| 128 | 1536 | bfloat16 | 4.361 | 2.425 | 3.645 | 2.676 |
| 512 | 1536 | bfloat16 | 9.741 | 3.675 | 5.813 | 7.178 |
| 2048 | 1536 | bfloat16 | 38.930 | 9.795 | 19.415 | 29.039 |

A fresh process confirmed the six tuning shapes. Five additional row counts were then measured without further kernel changes:

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 1536 | bfloat16 | 3.601 | 1.983 | 3.141 | 1.777 |
| 16 | 1536 | bfloat16 | 3.698 | 2.033 | 3.208 | 1.840 |
| 64 | 1536 | bfloat16 | 4.059 | 2.282 | 3.494 | 2.117 |
| 256 | 1536 | bfloat16 | 5.606 | 2.774 | 4.101 | 4.050 |
| 1024 | 1536 | bfloat16 | 19.509 | 5.792 | 10.626 | 14.548 |

Compiled reference remains faster on the smallest row counts; vectorization is not a universal win over every framework backend.

## Precision and width checks

The same four-method protocol was repeated for FP16/FP32 and additional BF16 widths. At widths other than 1536, the vector entry dispatches to the general warp kernel; their performance is shown explicitly. GPU replay medians in μs.

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1536 | float16 | 3.600 | 1.966 | 3.136 | 1.756 |
| 128 | 1536 | float16 | 4.408 | 2.361 | 3.630 | 2.642 |
| 2048 | 1536 | float16 | 39.949 | 9.922 | 19.843 | 29.024 |

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1536 | float32 | 3.495 | 2.090 | 3.471 | 1.888 |
| 128 | 1536 | float32 | 4.403 | 2.821 | 4.257 | 2.794 |
| 2048 | 1536 | float32 | 55.684 | 23.875 | 34.473 | 41.240 |

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 768 | bfloat16 | 2.984 | 3.207 | 2.683 | 1.439 |
| 128 | 768 | bfloat16 | 3.526 | 3.402 | 3.014 | 1.713 |
| 512 | 768 | bfloat16 | 7.302 | 4.162 | 3.996 | 3.494 |

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1537 | bfloat16 | 3.806 | 4.144 | 5.059 | 2.200 |
| 128 | 1537 | bfloat16 | 4.620 | 4.561 | 7.449 | 3.955 |
| 512 | 1537 | bfloat16 | 11.073 | 7.900 | 20.048 | 11.812 |

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 4096 | bfloat16 | 5.075 | 7.248 | 5.539 | 2.060 |
| 128 | 4096 | bfloat16 | 5.922 | 7.852 | 6.183 | 3.449 |
| 512 | 4096 | bfloat16 | 17.786 | 15.835 | 12.452 | 10.781 |

## Ordinary Python calls

Synchronized wall-time medians in μs from three interleaved loops of 100 calls, including Python dispatch, validation and output allocation. They are separate from graph replay latency.

| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1536 | bfloat16 | 6.366 | 5.357 | 11.714 | 32.955 |
| 8 | 1536 | bfloat16 | 5.657 | 5.074 | 11.556 | 31.161 |
| 32 | 1536 | bfloat16 | 4.451 | 4.196 | 7.475 | 22.986 |
| 128 | 1536 | bfloat16 | 5.149 | 4.359 | 8.038 | 22.010 |
| 512 | 1536 | bfloat16 | 11.191 | 4.530 | 10.183 | 29.040 |
| 2048 | 1536 | bfloat16 | 40.251 | 10.930 | 20.841 | 32.825 |

## Correctness and tooling

| Check | Result |
|---|---:|
| Full repository tests, including RMSNorm regression | 791 passed |
| LayerNorm tests across four variants | 311 passed |
| Independent CPU FP64-oracle numerical matrix | 720 passed |
| Memcheck, all API errors enabled, full suite | 791 tests; 0 errors |
| Synccheck, LayerNorm suite | 311 tests; 0 errors |
| Racecheck, LayerNorm suite | 311 tests; 0 hazards/errors/warnings |

The numerical matrix uses 10 shapes, three dtypes, four variants and six input modes: standard, zero, constant, tiny, large and shifted. Tolerances were fixed before tuning: FP32 rtol2e-5/atol5e-6; FP16 rtol/atol2e-3; BF16 rtol/atol2e-2. Tests cover tail widths, offset-but-contiguous storage, invalid contracts, non-default stream ordering and graph replay with changed inputs.

Explicit extension startup loading preserves the previously validated sanitizer initialization order. No sanitizer errors are suppressed. The NCU attempt again returned ERR_NVGPUCTRPERM; no host-driver permissions were changed. Static SASS confirms 20 versus 4 barrier sites and no local-memory instruction sites in the compiled LayerNorm kernels. Register/shared-memory counts are preserved in the resource dump; achieved occupancy/bandwidth is not inferred from them.

### Reference-path racecheck isolation

The first full LayerNorm racecheck run returned 24 analysis reports, all naming PyTorch FP64 MeanOps reduce_kernel, with no custom LayerNorm kernel in those reports. The independent oracle had been running FP64 means on the GPU. Its failure log remains in layernorm-racecheck.txt.

A PyTorch-only reproducer in diagnostics/torch_fp64_mean.py separates this path from the extension. The final oracle executes FP64 statistics and affine arithmetic on the CPU, then transfers the expected result for comparison. The same GPU operator cases remain checked with no name filters, exclusions or suppressions. CUDA kernel sources and all timing datasets are unchanged by the oracle change. The original and newer-tool control logs are retained; no vendor-internal cause is asserted.

## Evidence

- `layernorm-baseline.json` and `layernorm-baseline-profile.*`: initial source snapshot at commit 01e1068.
- `layernorm-tuning.json`, `layernorm-confirmation.json`, `layernorm-heldout.json`: ablation and independent confirmation.
- `layernorm-float16.json`, `layernorm-float32.json`, `layernorm-width-*.json`: dtype and fallback measurements.
- `layernorm-final-correctness.json`, `tests-layernorm-optimized.txt`, `layernorm-final-memcheck.txt`, `layernorm-final-synccheck.txt`, `layernorm-final-racecheck.txt`: numerical and execution checks.
- `layernorm-optimized-profile.json`, `layernorm-optimized-profile.resources.txt`, `layernorm-ncu.txt`: launch topology, static instruction/resource evidence and counter-access result.

## Conclusions

The width-1536 vectorized layout converts the reduction and memory-access changes into measured kernel latency gains. Small inputs still expose fixed launch/reduction costs, while non-specialized widths can favor native or compiled PyTorch. Full-model integration and GEMM are separate stages.

## References

- [PyTorch 2.8 LayerNorm semantics](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html)
- [PyTorch 2.8 native CUDA LayerNorm implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
- [CUDA 12.8 memory alignment](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#device-memory-accesses)
