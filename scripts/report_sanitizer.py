"""Validate recorded sanitizer evidence and write the initialization-order report."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / "results"
checks = [
    ("sanitizer-fixed-pytest", "Memcheck: complete test suite", "480 passed"),
    ("sanitizer-fixed-numerical", "Memcheck: numerical matrix", "PASS: 924 numerical cases"),
    ("sanitizer-fixed-alignment", "Memcheck: alignment and row tails", "PASS: 60 specialized"),
    ("sanitizer-fixed-all-api", "Memcheck: all API errors enabled", "PASS: 60 specialized"),
    ("sanitizer-fixed-synccheck", "Synccheck: alignment and row tails", "PASS: 60 specialized"),
    ("sanitizer-fixed-racecheck", "Racecheck: alignment and row tails", "PASS: 60 specialized"),
]
for stem, _, expected in checks:
    log = (results / f"{stem}.txt").read_text()
    assert (results / f"{stem}-exit.txt").read_text().strip() == "0", stem
    assert expected in log, stem
    summary = "0 hazards displayed (0 errors, 0 warnings)" if "racecheck" in stem else "ERROR SUMMARY: 0 errors"
    assert summary in log, stem
negative = (results / "sanitizer-negative-control.txt").read_text()
assert "Invalid __global__" in negative
assert (results / "sanitizer-negative-control-exit.txt").read_text().strip() == "99"
matrix = json.loads((results / "sanitizer-order/matrix.json").read_text())
assert [c["exit_code"] for c in matrix["cases"]] == [0, 0, 0, 99]
assert all(c["application_passed"] for c in matrix["cases"])
for name, digest in matrix["source_hashes"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
numerical = json.loads((results / "rmsnorm-sanitizer-correctness.json").read_text())
assert len(numerical["cases"]) == 924 and all(c["passed"] for c in numerical["cases"])
for name, digest in numerical["source_hashes"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

lines = [
    "# CUDA extension initialization-order diagnosis", "",
    "Explicit startup loading removes the recorded `cuKernelGetFunction` invalid-handle diagnostic from the L4 sanitizer workflow. CUDA kernels, numerical tolerances and API-error reporting remain unchanged.", "",
    "## Environment", "",
    "NVIDIA L4; driver 595.91.07; PyTorch 2.8.0+cu128; nvcc 12.8.93; loaded CUDA runtime package 12.8.90; Compute Sanitizer 2025.1.0.0. The process map records one loaded libcudart path from the PyTorch NVIDIA runtime package. Standalone `ldd` output is separate from the live Python process map.", "",
    "[Environment and loaded libraries](sanitizer-environment.txt). Isolated checks with Compute Sanitizer 2026.1.1 and 2026.3.0 also reproduced the late-load diagnostic; upgrading the checker alone did not resolve it. The system toolkit, driver and PyTorch installation were not replaced.", "",
    "## Controlled trigger", "",
    "Every row runs in a fresh process with the same add-scalar extension and numerical assertion. `torch.empty` allocates without filling; `torch.ones` launches a CUDA fill kernel.", "",
    "| Order before the first extension kernel | Memcheck | Exit code |",
    "|---|---:|---:|",
    "| Load extension, then execute CUDA tensor operations | 0 errors | 0 |",
    "| Initialize CUDA context, load extension, then execute operations | 0 errors | 0 |",
    "| Allocate an empty CUDA tensor, load extension, then fill | 0 errors | 0 |",
    "| Execute a PyTorch CUDA fill kernel, then load extension | 1 API error | 99 |", "",
    "[Commands, source hashes and matrix](sanitizer-order/matrix.json). All four applications passed their numerical assertions. A clean build in a separate directory reproduced the same order dependence: [early load](sanitizer-clean-build-old.txt), [late load](sanitizer-clean-lateload-old.txt).", "",
    "Standalone CUDA executables, a ctypes-loaded CUDA library, a minimal pybind JIT extension and a PyTorch-only workload passed their controls. Anonymous and templated standalone kernel controls were also clean. The trigger is established for the complete extension; this does not identify the internal NVIDIA driver/runtime mechanism or establish a universal CUDA late-loading defect.", "",
    "`CUDA_MODULE_LOADING=EAGER` alone did not resolve the original failure. That environment setting is different from loading this extension before CUDA kernel execution.", "",
    "## Startup change", "",
    "`gpu_ops.initialize()` completes native extension loading without launching a kernel. `scripts.sanitizer_workload` calls it before importing or running the selected CUDA workload. Normal operator entry points retain lazy loading; applications can explicitly initialize during startup.", "",
    "```python", "from gpu_ops import initialize", "initialize()", "```", "",
    "No suppression file, kernel exclusion, API-reporting reduction or warm-up launch is used. The native kernel sources are unchanged. Late loading can still reproduce the vendor-stack diagnostic; startup loading is the validated workaround, not a vendor-runtime patch.", "",
    "## L4 verification", "",
    "| Workload | Result | Exit code |", "|---|---|---:|",
]
for stem, label, expected in checks:
    count = "480 tests" if "pytest" in stem else ("924 cases" if "numerical" in stem else "60 cases")
    lines.append(f"| [{label}]({stem}.txt) | {count}; zero errors/hazards | 0 |")
lines += [
    "", "The numerical matrix covers seven RMSNorm variants, FP32/FP16/BF16, 11 shapes and four input magnitudes. The full pytest suite also checks invalid inputs, storage offsets, row tails, non-default streams and CUDA Graph replay.", "",
    "The separate [negative control](sanitizer-negative-control.txt) deliberately accesses index 32 of a one-float allocation. Memcheck reports invalid global accesses and exits 99, confirming that error detection remains active. Its source is `diagnostics/runtime_probe.cu`; it is not used by the operator library.", "",
    "## Check commands", "", "```bash",
    "python -m scripts.diagnose_sanitizer --output-dir results/new-order-matrix",
    "compute-sanitizer --tool memcheck --error-exitcode 99 python -m scripts.sanitizer_workload pytest",
    "compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 99 python -m scripts.sanitizer_workload alignment",
    "compute-sanitizer --tool memcheck --error-exitcode 99 python -m scripts.sanitizer_workload numerical --output results/new-numerical.json",
    "compute-sanitizer --tool synccheck --error-exitcode 99 python -m scripts.sanitizer_workload alignment",
    "compute-sanitizer --tool racecheck --error-exitcode 99 python -m scripts.sanitizer_workload alignment",
    "nvcc -arch=sm_89 -lineinfo diagnostics/runtime_probe.cu -o /tmp/runtime-probe",
    "compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/runtime-probe invalid",
    "```", "",
    "The last command is expected to fail with exit code 99. Use fresh output paths for new evidence.", "",
    "## References", "",
    "- [NVIDIA Compute Sanitizer: API errors, memory checks and synchronization checks](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)",
    "- [NVIDIA Compute Sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)",
    "- [Official standalone tool distributions](https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/)",
]
(results / "sanitizer-diagnosis.md").write_text("\n".join(lines) + "\n")
print("Validated sanitizer evidence and generated results/sanitizer-diagnosis.md")
