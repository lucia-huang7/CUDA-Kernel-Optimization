"""Population-variance LayerNorm with FP32 statistics and affine arithmetic."""
import torch
import torch.nn.functional as F


def layernorm_reference(x, weight, bias, eps=1e-5):
    values = x.float()
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(variance + eps) * weight.float() + bias.float()).to(x.dtype)


def layernorm_native(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
