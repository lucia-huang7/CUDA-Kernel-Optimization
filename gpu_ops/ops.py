"""Forward-only CUDA operators exposed by the pybind extension."""
from .extension import load_extension


def add_scalar(x, value=1.0):
    return load_extension().add_scalar(x, value)


def rmsnorm(x, weight, eps=1e-6):
    return load_extension().rmsnorm(x, weight, eps)


def rmsnorm_warp(x, weight, eps=1e-6):
    return load_extension().rmsnorm_warp(x, weight, eps)


def rmsnorm_cached(x, weight, eps=1e-6):
    return load_extension().rmsnorm_cached(x, weight, eps)


def rmsnorm_rows(x, weight, eps=1e-6):
    return load_extension().rmsnorm_rows(x, weight, eps)


def rmsnorm_vector(x, weight, eps=1e-6):
    return load_extension().rmsnorm_vector(x, weight, eps)


def rmsnorm_reload(x, weight, eps=1e-6):
    return load_extension().rmsnorm_reload(x, weight, eps)


def rmsnorm_vector128(x, weight, eps=1e-6):
    return load_extension().rmsnorm_vector128(x, weight, eps)


def layernorm(x, weight, bias, eps=1e-5):
    return load_extension().layernorm(x, weight, bias, eps)


def layernorm_warp(x, weight, bias, eps=1e-5):
    return load_extension().layernorm_warp(x, weight, bias, eps)


def layernorm_cached(x, weight, bias, eps=1e-5):
    return load_extension().layernorm_cached(x, weight, bias, eps)


def layernorm_vector(x, weight, bias, eps=1e-5):
    return load_extension().layernorm_vector(x, weight, bias, eps)
