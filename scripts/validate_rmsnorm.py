"""Save numerical errors for representative shapes and input magnitudes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from gpu_ops import rmsnorm, rmsnorm_warp, rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128
from reference.rmsnorm import rmsnorm_reference

TOLERANCES = {
    torch.float32: {'rtol': 1e-5, 'atol': 2e-6},
    torch.float16: {'rtol': 2e-3, 'atol': 2e-3},
    torch.bfloat16: {'rtol': 2e-2, 'atol': 2e-2},
}


def run(output):
    output = Path(output)
    if output.exists():
        raise SystemExit("Choose a new output filename.")
    torch.manual_seed(20261015)
    records = []
    for dtype, tolerance in TOLERANCES.items():
        for rows, width in [(1, 1), (1, 1536), (8, 1537), (32, 256), (128, 768), (512, 1536), (2048, 1536), (2, 65536), (3, 1536), (5, 1536), (7, 1536)]:
            for scale in [0.0, 1e-7, 1.0, 1e4]:
                x = torch.randn(rows, width, device='cuda', dtype=dtype) * scale
                weight = torch.randn(width, device='cuda', dtype=dtype)
                expected = rmsnorm_reference(x, weight)
                for variant, operation in [('shared', rmsnorm), ('warp', rmsnorm_warp), ('cached', rmsnorm_cached), ('rows', rmsnorm_rows), ('vector', rmsnorm_vector), ('reload', rmsnorm_reload), ('vector128', rmsnorm_vector128)]:
                    actual = operation(x, weight)
                    torch.testing.assert_close(actual, expected, **tolerance)
                    errors = (actual.float() - expected.float()).abs()
                    records.append({'variant': variant, 'dtype': str(dtype), 'rows': rows,
                                    'width': width, 'scale': scale, 'tolerance': tolerance,
                                    'max_abs_error': errors.max().item(),
                                    'max_rel_error_floor_1e_3': (errors / expected.float().abs().clamp_min(1e-3)).max().item(),
                                    'passed': True})
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
              ['csrc/bindings.cpp', 'csrc/kernels.cu', 'reference/rmsnorm.py', 'scripts/validate_rmsnorm.py', 'csrc/rmsnorm_specialized.cu', 'gpu_ops/extension.py']}
    payload = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
               'seed': 20261015, 'eps': 1e-6, 'source_hashes': hashes, 'cases': records}
    Path('results').mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'PASS: {len(records)} numerical cases across FP32, FP16 and BF16.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/rmsnorm-specialized-final-correctness.json")
    run(parser.parse_args().output)
