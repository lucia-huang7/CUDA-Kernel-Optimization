"""Record a fresh-process initialization-order matrix without suppressions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SETUPS = {
    "preload": "m=load_extension();x=torch.ones(257,device='cuda');",
    "context_first": "torch.cuda.init();m=load_extension();x=torch.ones(257,device='cuda');",
    "allocation_first": "x=torch.empty(257,device='cuda');m=load_extension();x.fill_(1);",
    "kernel_first": "x=torch.ones(257,device='cuda');m=load_extension();",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitizer", default="compute-sanitizer")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    version = subprocess.run([args.sanitizer, "--version"], capture_output=True, text=True, check=True)
    for name, setup in SETUPS.items():
        code = ("import torch;from gpu_ops.extension import load_extension;" + setup +
                "torch.testing.assert_close(m.add_scalar(x,1),x+1);torch.cuda.synchronize();print('APPLICATION_PASSED')")
        command = [args.sanitizer, "--tool", "memcheck", "--error-exitcode", "99", sys.executable, "-c", code]
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / f"{name}.txt").write_text(run.stdout)
        records.append({"case": name, "command": command, "exit_code": run.returncode,
                        "application_passed": "APPLICATION_PASSED" in run.stdout,
                        "clean_memcheck": "ERROR SUMMARY: 0 errors" in run.stdout and run.returncode == 0})
        print(f"{name}: exit={run.returncode}, clean={records[-1]['clean_memcheck']}", flush=True)
    sources = ["csrc/bindings.cpp", "csrc/kernels.cu", "csrc/rmsnorm_specialized.cu",
               "gpu_ops/extension.py", "scripts/diagnose_sanitizer.py"]
    payload = {"sanitizer_version": version.stdout + version.stderr,
               "module_loading": os.environ.get("CUDA_MODULE_LOADING", "default"),
               "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
               "cases": records}
    (output / "matrix.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
