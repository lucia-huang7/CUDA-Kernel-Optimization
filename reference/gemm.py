"""FP32 PyTorch GEMM control and independent CPU FP64 oracle."""
import torch


def configure_precision():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False


def gemm_reference(a,b):
    return torch.mm(a,b)


def gemm_oracle(a,b):
    return (a.cpu().double() @ b.cpu().double()).float().to(a.device)


def tolerance(k):
    return dict(rtol=2e-4,atol=2e-3) if k>512 else dict(rtol=5e-5,atol=5e-5)
