# GPU Kernel Lab

CUDA C++ operator development with PyTorch bindings, numerical tests and hardware profiling. The project covers normalization and matrix multiplication, with implementations and measurements added in stages.

## Implementation

| Component | Source |
|---|---|
| Python entry points | [gpu_ops/ops.py](gpu_ops/ops.py) |
| Extension loader | [gpu_ops/extension.py](gpu_ops/extension.py) |
| Pybind interface | [csrc/bindings.cpp](csrc/bindings.cpp) |
| CUDA kernels and launch wrappers | [csrc/kernels.cu](csrc/kernels.cu) |
| RMSNorm reference | [reference/rmsnorm.py](reference/rmsnorm.py) |

The initial RMSNorm kernel uses one 256-thread block per row, FP32 reduction and Qwen2-compatible cast order. It supports contiguous two-dimensional FP32, FP16 and BF16 tensors. [Development workflow](docs/DEVELOPMENT.md).
