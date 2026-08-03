"""Compile the extension and validate Python-to-CUDA execution."""
import json
import time
from pathlib import Path
import torch
from gpu_ops import add_scalar
from gpu_ops.extension import load_extension


def run():
    start = time.perf_counter()
    module = load_extension()
    build_seconds = time.perf_counter() - start
    checks = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for count in [0, 1, 255, 256, 257, 4097]:
            x = torch.arange(count, device="cuda", dtype=torch.float32).remainder(127).to(dtype)
            actual = add_scalar(x, 1.0)
            torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)
            checks.append({"dtype": str(dtype), "count": count, "passed": True})
    torch.cuda.synchronize()
    result = {"extension": Path(module.__file__).name, "load_seconds": build_seconds,
              "gpu": torch.cuda.get_device_name(), "checks": checks, "passed": True}
    Path("results").mkdir(exist_ok=True)
    Path("results/extension-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
