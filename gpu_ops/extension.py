"""Build and load the CUDA extension through PyTorch's C++ extension loader."""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[1]
    build = root / ".build"
    build.mkdir(exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name="gpu_ops_cuda",
        sources=[str(root / "csrc/bindings.cpp"), str(root / "csrc/kernels.cu"),
                 str(root / "csrc/rmsnorm_specialized.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
        build_directory=str(build),
        verbose=os.environ.get("GPU_OPS_BUILD_VERBOSE", "0") == "1",
    )


def initialize():
    """Load native kernels during startup, before the first CUDA tensor operation.

    Explicit startup loading avoids the late-load cuKernelGetFunction diagnostic
    observed with CUDA 12.8 on the recorded L4 environment. No kernel is launched.
    """
    load_extension()
