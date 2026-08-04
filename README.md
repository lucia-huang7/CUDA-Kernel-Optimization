# GPU Kernel Lab

CUDA C++ operator development with PyTorch bindings, numerical tests and hardware profiling. The project covers normalization and matrix multiplication, with implementations and measurements added in stages.

## Validation

**NVIDIA L4 · PyTorch 2.8.0 · CUDA 12.8 · FP32 / FP16 / BF16**

| Check | Result |
|---|---:|
| CUDA extension compile and first load | 17.49 s |
| Python-to-CUDA smoke cases | 18 passed |
| Correctness and execution-semantics tests | 51 passed |
| Separate RMSNorm numerical cases | 96 passed |

[Validation results](results/validation.md) · [Raw numerical evidence](results/rmsnorm-correctness.json)

## Implementation

| Component | Source |
|---|---|
| Python entry points | [gpu_ops/ops.py](gpu_ops/ops.py) |
| Extension loader | [gpu_ops/extension.py](gpu_ops/extension.py) |
| Pybind interface | [csrc/bindings.cpp](csrc/bindings.cpp) |
| CUDA kernels and launch wrappers | [csrc/kernels.cu](csrc/kernels.cu) |
| RMSNorm reference | [reference/rmsnorm.py](reference/rmsnorm.py) |

The initial RMSNorm kernel uses one 256-thread block per row, FP32 reduction and Qwen2-compatible cast order. It supports contiguous two-dimensional FP32, FP16 and BF16 tensors. [Development workflow](docs/DEVELOPMENT.md).
