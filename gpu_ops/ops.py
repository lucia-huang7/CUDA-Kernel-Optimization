"""Forward-only CUDA operators exposed by the pybind extension."""
from .extension import load_extension


def add_scalar(x, value=1.0):
    return load_extension().add_scalar(x, value)


def rmsnorm(x, weight, eps=1e-6):
    return load_extension().rmsnorm(x, weight, eps)


def rmsnorm_warp(x, weight, eps=1e-6):
    return load_extension().rmsnorm_warp(x, weight, eps)
