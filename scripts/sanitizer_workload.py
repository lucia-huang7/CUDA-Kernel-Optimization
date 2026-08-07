"""Run CUDA checks with native registration completed before CUDA workloads."""
import argparse
import runpy
import sys

from gpu_ops import initialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=["alignment", "pytest", "numerical"])
    parser.add_argument("--output", help="Fresh JSON path for the numerical workload")
    args = parser.parse_args()
    if args.workload == "numerical" and not args.output:
        parser.error("numerical requires --output")

    initialize()
    print("Native extension loaded before CUDA workloads.", flush=True)
    if args.workload == "alignment":
        runpy.run_module("scripts.sanitize_specialized", run_name="__main__")
    elif args.workload == "pytest":
        import pytest
        raise SystemExit(pytest.main(["-q", "tests"]))
    else:
        sys.argv = ["validate_rmsnorm", "--output", args.output]
        runpy.run_module("scripts.validate_rmsnorm", run_name="__main__")


if __name__ == "__main__":
    main()
