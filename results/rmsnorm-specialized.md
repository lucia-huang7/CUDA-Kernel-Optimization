# RMSNorm Register and Vector Layout Experiment

NVIDIA L4, BF16, width 1536. Primary results use a fresh-process confirmation run after candidate evaluation. Row counts 2, 16, 64, 256 and 1024 were measured separately after tuning.

## Diagnosis and changes

Generated Inductor code also contains two input load sites. That observation does not make repeated loads the sole explanation for the earlier gap. The prior CUDA warp kernel uses runtime width, a 256-thread row block and scalar loads. This experiment separates fixed-width reload, register reuse, row grouping and vectorized layouts.

| Variant | Row mapping | Input retention and access |
|---|---|---|
| cuda_warp | 256 threads per row | Runtime-width loops, input reread |
| cuda_reload | 256 threads per row | Width-1536 unrolled loops, input reread control |
| cuda_cached | 256 threads per row | Width-1536 unrolled loops, six FP32 values retained per thread |
| cuda_rows | One warp per row, four rows per block | 48 FP32 values retained per thread, scalar access |
| cuda_vector | One warp per row, four rows per block | Retained values and aligned 16-byte packs |
| cuda_vector128 | 128 threads per row | Twelve FP32 values retained per thread, four-element packs |

The selected vector128 path uses 8-byte loads/stores for BF16/FP16 and 16-byte loads/stores for FP32. Two block-wide barriers remain. Widths other than 1536 use the previous warp implementation. Misaligned contiguous storage falls back to a scalar-safe cached path; no runtime timing-based autotuning or row-count lookup is used.

## Confirmation run

Each value is the median CUDA event time per invocation in a 100-call graph replay, across 30 randomized method-order repetitions. Inputs and weights are reused. Compilation, Python dispatch and allocation setup are outside replay timing.

| Rows | Previous CUDA (us) | Cached 256 (us) | Vector 128 (us) | PyTorch compiled (us) | Latency reduction vs previous |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.009 | 2.075 | 1.674 | 1.426 | 44.4% |
| 8 | 3.104 | 2.132 | 1.717 | 1.444 | 44.7% |
| 32 | 3.137 | 2.167 | 1.821 | 1.556 | 41.9% |
| 128 | 3.309 | 2.373 | 2.070 | 1.897 | 37.4% |
| 512 | 6.303 | 4.323 | 3.044 | 4.293 | 51.7% |
| 2048 | 22.379 | 13.398 | 9.359 | 17.905 | 58.2% |

## Held-out row counts

These shapes were not used in the candidate tuning run. Width, dtype, graph timing and numerical tolerance remain unchanged.

| Rows | Previous CUDA (us) | Vector 128 (us) | PyTorch compiled (us) | Vector / compiled latency |
|---:|---:|---:|---:|---:|
| 2 | 3.008 | 1.669 | 1.414 | 1.18x |
| 16 | 3.110 | 1.731 | 1.477 | 1.17x |
| 64 | 3.207 | 1.961 | 1.692 | 1.16x |
| 256 | 3.565 | 2.357 | 2.804 | 0.84x |
| 1024 | 10.693 | 5.229 | 8.243 | 0.63x |

## Ablation controls

The following data comes from the candidate/control run, separate from the confirmation run. The scalar multi-row candidate regressed; it remains selectable and its measurements are preserved.

| Rows | Runtime warp | Fixed reload | Cached 256 | Scalar rows | Vector rows | Vector 128 | Compiled |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.014 | 2.141 | 2.085 | 8.022 | 2.280 | 1.679 | 1.433 |
| 8 | 3.106 | 2.200 | 2.138 | 8.109 | 2.468 | 1.725 | 1.448 |
| 32 | 3.146 | 2.246 | 2.185 | 8.495 | 2.572 | 1.821 | 1.565 |
| 128 | 3.288 | 2.448 | 2.399 | 8.472 | 2.642 | 2.069 | 1.892 |
| 512 | 6.584 | 4.856 | 4.522 | 9.277 | 4.133 | 3.106 | 4.523 |
| 2048 | 24.123 | 17.131 | 14.562 | 23.370 | 15.123 | 10.216 | 19.384 |

All entries are microseconds. Fixed-width specialization changes loop/address code as well as allowing register retention. Compare fixed reload with cached to isolate retention under the same width and block mapping; compare runtime warp with fixed reload for the specialization effect. The vector128 candidate changes thread mapping and vector width together, so those contributions are not independently isolated.

## Ordinary Python API calls

Median of three synchronized 100-call loops, including dispatch, input validation and output allocation. This is not model or HTTP latency.

| Rows | Previous CUDA (us) | Vector 128 (us) | PyTorch compiled (us) |
|---:|---:|---:|---:|
| 1 | 3.767 | 3.851 | 20.100 |
| 8 | 4.596 | 5.880 | 27.297 |
| 32 | 4.354 | 4.510 | 23.419 |
| 128 | 4.086 | 3.630 | 20.014 |
| 512 | 7.722 | 4.233 | 19.984 |
| 2048 | 23.504 | 10.497 | 20.715 |

## Correctness and compiled evidence

All 480 tests passed. The separate sweep passed 924 cases across seven variants and FP32/FP16/BF16. Tests include row tails, offset-but-contiguous input/weight storage, unsupported shapes/layouts, CUDA stream ordering and graph replay. All paths preserve the normalized-activation cast before weight multiplication.

The compiled resource/disassembly records retain vector instruction sites, barrier counts, local-memory instruction sites and register/shared-storage usage. These are static compiler evidence, not achieved bandwidth, occupancy or runtime stall measurements. The previous NCU attempt was denied with ERR_NVGPUCTRPERM; no driver permissions were changed.

Selected BF16/FP16 SASS uses six 64-bit global-load sites and three 64-bit store sites; FP32 uses the corresponding 128-bit instructions. The selected BF16 kernel uses 28 registers per thread and 16 bytes of static shared storage, with no local-memory instruction sites. These are compiled resource observations rather than achieved occupancy measurements.

## Evidence files

- `rmsnorm-specialized-tuning.json`: first candidate exploration.
- `rmsnorm-specialized-controls.json`: specialization/reload/cache/layout controls.
- `rmsnorm-specialized-confirmation.json`: primary fresh-process results.
- `rmsnorm-specialized-heldout.json`: additional row counts.
- `rmsnorm-specialized-final-correctness.json`, `tests-specialized-final.txt`: numerical and execution-semantics checks.
- `compiled-rmsnorm-inspection.json`: generated Inductor snippets.
- `rmsnorm-specialized-binary.json`, `rmsnorm-specialized-resources.txt`: static binary evidence.

The optimization targets width 1536. Other widths retain the tested fallback; they are not claimed faster. PyTorch compiled remains a stronger control than eager execution and can remain faster on small shapes. Full-model benefit is a separate experiment.

## Sanitizer diagnostics

Explicit startup loading resolves the recorded initialization-order diagnostic in the sanitizer workflow. The L4 memcheck run passes all 480 tests and the 60 alignment/tail comparisons with zero errors. Historical late-load failures remain available.

See [initialization-order diagnosis](sanitizer-diagnosis.md) for the controlled reproducer, numerical checks and negative control. No CUDA kernel or API-error reporting mode was changed.
