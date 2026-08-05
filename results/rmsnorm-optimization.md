# RMSNorm Warp Reduction Experiment

NVIDIA L4; BF16; width 1536; one 256-thread block per row. The CUDA variants differ only in their block reduction: a shared-memory tree versus warp shuffles and eight shared partial sums.

## GPU replay latency

The table reports median CUDA event time per invocation in a 100-call CUDA Graph sequence, with 30 interleaved repeats. Compilation, Python dispatch and output-allocation setup are outside replay timing. Inputs and weights are reused; this is a warm-cache microbenchmark.

| Rows | PyTorch eager (us) | CUDA shared (us) | CUDA warp (us) | PyTorch compiled (us) | Shared / warp |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.197 | 3.183 | 3.012 | 1.432 | 1.06x |
| 8 | 11.925 | 3.268 | 3.097 | 1.460 | 1.06x |
| 32 | 13.391 | 3.304 | 3.140 | 1.564 | 1.05x |
| 128 | 15.375 | 3.616 | 3.288 | 1.897 | 1.10x |
| 512 | 30.805 | 7.528 | 6.587 | 4.641 | 1.14x |
| 2048 | 130.632 | 29.852 | 25.579 | 20.631 | 1.17x |

## Ordinary Python API wall time

Each value is the median of three synchronized 100-call loops, including Python validation/dispatch and output allocation. These are operator API calls, not complete model or HTTP request timings.

| Rows | CUDA shared (us) | CUDA warp (us) | Shared / warp |
|---:|---:|---:|---:|
| 1 | 3.949 | 3.774 | 1.05x |
| 8 | 4.043 | 3.891 | 1.04x |
| 32 | 4.077 | 3.909 | 1.04x |
| 128 | 4.334 | 4.068 | 1.07x |
| 512 | 8.340 | 7.582 | 1.10x |
| 2048 | 32.562 | 28.073 | 1.16x |

## Mechanism and correctness

Baseline source executes nine block-wide barriers: one initial publication plus eight shared-tree stages. Warp reduction uses five shuffle steps per warp, publishes eight warp sums, reduces them in the first warp, then publishes the total with two block-wide barriers. Input loads, row mapping, block size, output allocation and precision semantics remain unchanged.

Compiled instruction evidence is recorded below. Counts are static SASS instructions, not stall metrics or measured dynamic instruction totals.

| Variant | Dtype | Static BAR.SYNC | Static SHFL |
|---|---|---:|---:|
| warp | bfloat16 | 2 | 11 |
| warp | float16 | 2 | 11 |
| warp | float32 | 2 | 11 |
| shared | bfloat16 | 9 | 0 |
| shared | float16 | 9 | 0 |
| shared | float32 | 9 | 0 |

All 98 tests passed. The separate sweep passed 192 numerical cases across both variants, FP32/FP16/BF16, tails, extreme tested magnitudes and large widths. CUDA stream ordering, graph replay and contract rejection are tested for both variants.

## Profiling access

Nsight Compute 2025.1.1 was invoked with SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats and MemoryWorkloadAnalysis. The cloud driver denied counter access with ERR_NVGPUCTRPERM; the original output and exit status are retained. No stall, achieved bandwidth, achieved occupancy or utilization percentage is claimed. Static resource/disassembly inspection and unprofiled timing support a controlled reduction experiment; they do not establish that barriers are the only bottleneck.

## Evidence

- `rmsnorm-baseline.json`: timing before the warp variant was added.
- `rmsnorm-warp-comparison.json`: simultaneous reference/shared/warp/compiled controls and raw samples.
- `rmsnorm-warp-correctness.json`, `tests-warp.txt`: numerical and execution-semantics validation.
- `rmsnorm-binary-evidence.json`, `rmsnorm-binary-resources.txt`: compiled instruction and resource evidence.
- `ncu-shared-2048.csv`, `ncu-shared-exit.txt`: hardware-counter access failure.

The compiled PyTorch control is kept separate from the eager reference. Operator results do not establish model-level speedup. LayerNorm and GEMM remain subsequent stages.
