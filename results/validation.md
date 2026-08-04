# CUDA Extension and RMSNorm Validation

GPU: NVIDIA L4. PyTorch: 2.8.0+cu128. CUDA: 12.8.

The C++/CUDA extension compiled and loaded in 17.49 seconds on the recorded first call. All 18 add-scalar smoke cases passed.

The L4 test suite passed 51 tests, including shape/dtype handling, rejected inputs, non-default stream ordering and CUDA Graph replay with updated inputs.

The separate RMSNorm sweep passed 96 cases. It covers FP32, FP16 and BF16; widths 1–65536; 1–2048 rows; and input scales 0, 1e-7, 1 and 1e4.

| Dtype | Cases | Maximum absolute error | Maximum relative error, denominator floor 1e-3 |
|---|---:|---:|---:|
| float32 | 32 | 1.90735e-06 | 3.42404e-07 |
| float16 | 32 | 0.00195312 | 0.00152207 |
| bfloat16 | 32 | 0.0078125 | 0.0107527 |

Correctness uses fixed elementwise tolerances stored per case. Aggregate maximum relative error alone is not the pass condition. Performance tuning and GEMM are subsequent stages.
