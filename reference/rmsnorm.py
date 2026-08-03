"""Qwen2 RMSNorm semantics: FP32 reduction and rounding before weight scaling."""
import torch


def rmsnorm_reference(x, weight, eps=1e-6):
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps)
    return normalized.to(x.dtype) * weight
