# FP32 GEMM: tiling, register reuse and small-M parallelism

The CUDA GEMM implementation progresses from one output per thread to shared-memory tiles, register blocking, four-element vector access and guarded split-K. Independent L4 confirmation measures **2.34×** and **3.79×** speedups over the CUDA baseline on the two target rectangular shapes. PyTorch and direct cuBLAS remain separate precision-controlled comparisons.

## Contract and numerical controls

`C = A @ B`, with A[M,K], B[K,N] and C[M,N] in contiguous row-major FP32 storage. The operators are forward-only, use the current CUDA stream, reject autograd inputs, and handle empty output dimensions and K=0. Misaligned contiguous storage is accepted through scalar fallback paths. Dimensions must fit the integer API and the supported row grid; M is capped at 65535×16.

The custom kernels explicitly use FP32 fmaf accumulation. Existing normalization compiler options are unchanged. PyTorch uses highest float32 matmul precision with TF32 disabled. The direct cuBLAS wrapper uses cublasGemmEx with CUBLAS_COMPUTE_32F_PEDANTIC, alpha=1 and beta=0, reversing A/B dimensions for column-major cuBLAS. Pointer mode is restored after the call. This experiment measures FP32 SIMT optimization; it does not claim Tensor Core or low-precision GEMM performance.

## Locate and change

The initial baseline passed 860 full-repository tests. At M16/N4096/K4096, the first run measured 614.834 μs versus direct cuBLAS 284.192 μs. At M4096/N4096/K32, it measured 1007.178 μs versus 270.623 μs.

| Stage | Reuse or parallelism change |
|---|---|
| Naive | One thread computes one output; global loads inside the K loop |
| Shared tile | Cooperatively load 16×16 tiles of A and B; two barriers per K tile |
| Register tile | Retain 8 or 16 output accumulators per thread and reuse A/B fragments |
| Vector tile | Four-element global/shared loads and output stores; alignment guards |
| Small-M split-K | Compute four K partitions in parallel, then reduce their partial outputs |

The register candidates vary BM/BN/BK/TM/TN: 64/64/16/4/4, 32/64/16/2/4, 16/64/32/2/4, 16/128/32/2/8, 32/128/32/4/4 and 64/128/32/4/4. The source example tile is one candidate; its performance is remeasured on L4. Three vector layouts and split counts 4/8 are compared separately.

Register blocking alone regressed on the small-M target. Vector access improved the small-K path, while split-K improved the long-K, low-row-count path. For the latter, the 16×64 tile exposes 64 blocks; four splits expose 256 compute blocks plus a reduction kernel. The four-part workspace is 1 MiB at the target small-M shape.

The final small-M profile identifies a SIMT SGEMM kernel plus splitKreduce_kernel in both PyTorch and direct cuBLAS. Tuned CUDA likewise uses its vectorized partial-product kernel and a separate reduction. These are whole-operation comparisons, including both kernels.

### Selected dispatch

- M≤64 and K≥256: split-K with four partitions, subject to vector alignment/divisibility guards.
- M≥128 and K≤64: 32×128 vector tile.
- Other aligned shapes: 16×64 vector tile.
- Unsupported vector alignment or N/K divisibility: general scalar register tile.

The guards preserve correctness; timing for an unaligned-width fallback is included below. Selection uses the tuning run only; the held-out shapes do not change these rules.

## Independent confirmation

FP32 GPU replay medians in μs. Twenty calls per capture, 30 randomized interleaved repetitions, 20 warm-up calls and five warm-up replays. First calls, including setup, are recorded separately. Inputs are fixed across repeats, so the measurements are not cold-memory or streaming-bandwidth tests.

![GEMM performance](../assets/figures/gemm_v1.png)

| M | N | K | CUDA baseline | Shared tile | Tuned CUDA | PyTorch | Direct cuBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 4096 | 617.044 | 464.093 | 263.518 | 281.820 | 282.527 |
| 4096 | 4096 | 32 | 1004.693 | 791.953 | 265.044 | 274.468 | 271.799 |
| 512 | 512 | 512 | 191.522 | 149.533 | 68.668 | 32.454 | 32.423 |

Effective arithmetic throughput uses 2MNK / elapsed time; it is not a hardware utilization counter.

| Shape M×N×K | Tuned TFLOP/s | Direct cuBLAS TFLOP/s | Baseline / tuned |
|---|---:|---:|---:|
| 16×4096×4096 | 2.037 | 1.900 | 2.34× |
| 4096×4096×32 | 4.051 | 3.950 | 3.79× |
| 512×512×512 | 3.909 | 8.279 | 2.79× |

### Sample spread

Tuned and cuBLAS interquartile ranges from the same 30-repeat runs, in μs. Small differences should be assessed with this spread and the held-out results.

| Shape | Tuned median [Q1,Q3] | cuBLAS median [Q1,Q3] |
|---|---:|---:|
| 16×4096×4096 | 263.518 [263.070, 263.872] | 282.527 [282.110, 283.723] |
| 4096×4096×32 | 265.044 [264.487, 265.683] | 271.799 [270.786, 275.066] |
| 512×512×512 | 68.668 [67.882, 68.911] | 32.423 [32.054, 32.574] |

## Held-out shapes

These five shapes were measured after candidate selection with no further tuning. GPU replay medians in μs.

| M | N | K | CUDA baseline | Shared tile | Tuned CUDA | PyTorch | Direct cuBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2048 | 2048 | 67.146 | 100.719 | 33.964 | 30.237 | 30.675 |
| 32 | 2048 | 1024 | 110.424 | 88.388 | 32.459 | 19.126 | 19.144 |
| 2048 | 3072 | 32 | 382.721 | 299.919 | 97.819 | 97.015 | 98.619 |
| 1024 | 2048 | 64 | 206.117 | 159.851 | 45.315 | 37.432 | 37.370 |
| 257 | 511 | 129 | 22.359 | 20.602 | 19.529 | 8.246 | 8.269 |

The tuned entry improves on the naive baseline on these shapes, while cuBLAS remains faster on multiple held-out shapes and the square control. The rectangular target results are not a general replacement claim.

## Ordinary Python calls

Synchronized wall-time medians in μs, three interleaved loops of 20 calls, including validation, dispatch, workspace allocation and output allocation.

| M | N | K | CUDA baseline | Shared tile | Tuned CUDA | PyTorch | Direct cuBLAS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 4096 | 618.413 | 461.253 | 266.459 | 285.727 | 285.741 |
| 4096 | 4096 | 32 | 1019.827 | 793.211 | 266.564 | 268.328 | 266.046 |
| 512 | 512 | 512 | 193.308 | 150.948 | 70.519 | 35.599 | 35.499 |

## Candidate evidence

All candidate measurements remain in the raw tuning files, including slower configurations. These stages have distinct source hashes and Git snapshots; do not combine their medians into a paired speedup.

| Dataset | Source revision | Candidate stage |
|---|---|---|
| [gemm-baseline.json](gemm-baseline.json) | be4352c | Naive / PyTorch / cuBLAS |
| [gemm-tuning.json](gemm-tuning.json) | abbd252 | Shared and six register layouts |
| [gemm-vector-tuning.json](gemm-vector-tuning.json) | cb24660 | Three vector layouts |
| [gemm-split-tuning.json](gemm-split-tuning.json) | cc0fa3e | Split count 4 / 8 |

## Memory model and profiling

Unique input/output storage gives a lower-bound traffic model of 4(MK+KN+MN) bytes and arithmetic intensity 2MNK / bytes. It assumes one read of each input and one output write; shared-tile rereads, split workspace, cache behavior and write traffic can change actual transfers.

| M×N×K | Operations | Unique I/O bytes | Intensity (FLOP/byte) |
|---|---:|---:|---:|
| 16×4096×4096 | 536,870,912 | 67,633,152 | 7.94 |
| 4096×4096×32 | 1,073,741,824 | 68,157,440 | 15.75 |
| 512×512×512 | 268,435,456 | 3,145,728 | 85.33 |

The GPU has 58 SMs. Launch topology, static SASS and compiler resources are recorded in gemm-final-profile.json and its resource dump. NCU counter access again failed with ERR_NVGPUCTRPERM; no achieved occupancy/bandwidth or bank-conflict reduction is claimed from static evidence.

### Compiled resources

| Kernel/layout | Registers per thread | Shared bytes/block | Local bytes |
|---|---:|---:|---:|
| Naive | 40 | 0 | 0 |
| Shared 16×16 | 38 | 2048 | 0 |
| Vector 16×64 | 56 | 10304 | 0 |
| Vector 32×128 | 72 | 20608 | 0 |
| Small-M split-K compute | 64 | 10304 | 0 |
| Split reduction | 36 | 0 | 0 |

These values are compiler resource requirements, not achieved occupancy. The recorded custom kernel disassembly has no local-memory instruction sites.

## Correctness and tooling

| Check | Result |
|---|---:|
| Full repository tests | 1330 passed |
| GEMM-specific tests | 539 passed |
| Numerical cases against CPU FP64 oracle | 768 passed |
| Memcheck, all API errors enabled | 1330 tests; 0 errors |
| Synccheck, all candidate boundary workload | 192 cases; 0 errors |
| Racecheck, all candidate boundary workload | 192 cases; 0 hazards/errors/warnings |

The numerical matrix covers eight shapes, six input modes and 16 entries including PyTorch, direct cuBLAS, every candidate and tuned dispatch. Modes include zeros, constants, tiny/large magnitudes and cancellation. Fixed tolerances are rtol5e-5/atol5e-5 for K≤512 and rtol2e-4/atol2e-3 for longer K; the large-magnitude case scales absolute tolerance by the squared input scale. Every error and tolerance is retained in gemm-correctness.json.

Tests cover row/column/K tails, empty dimensions, storage offsets, invalid contracts/configurations, non-default streams and CUDA Graph replay after changing inputs. The compact synchronization/race workload exercises all 16 entries on tail/alignment cases and empty dimensions. The independent oracle runs FP64 arithmetic on the CPU; no sanitizer name filters or suppressions are used.

## Sources

- [CUDA shared-memory GEMM guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
- [cuBLAS precision modes and GEMM API](https://docs.nvidia.com/cuda/cublas/index.html)
- [PyTorch CUDA numerical precision](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
