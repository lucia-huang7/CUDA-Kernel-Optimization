"""Record compiler and GPU environment for extension validation."""
import json
import platform
import shutil
import subprocess
import torch


def collect_environment():
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI, "tools": {}}
    for name in ["nvcc", "g++", "ninja", "ncu", "nsys"]:
        path = shutil.which(name)
        entry = {"available": path is not None}
        if path:
            try:
                entry["version"] = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT, timeout=15).strip()
            except subprocess.SubprocessError as error:
                entry["error"] = type(error).__name__
        result["tools"][name] = entry
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {"name": gpu.name, "memory_bytes": gpu.total_memory,
                         "compute_capability": list(torch.cuda.get_device_capability())}
    return result


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
