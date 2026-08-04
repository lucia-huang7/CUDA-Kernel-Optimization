"""Summarize saved CUDA extension and RMSNorm correctness evidence."""
import json
from pathlib import Path


def report(root=Path('results')):
    smoke = json.loads((root / 'extension-smoke.json').read_text())
    numerical = json.loads((root / 'rmsnorm-correctness.json').read_text())
    environment = json.loads((root / 'environment.json').read_text())
    tests = (root / 'tests.txt').read_text().strip()
    assert smoke['passed'] and all(case['passed'] for case in numerical['cases'])
    assert '51 passed' in tests and 'failed' not in tests
    lines = ['# CUDA Extension and RMSNorm Validation', '',
             f"GPU: {smoke['gpu']}. PyTorch: {environment['torch']}. CUDA: {environment['torch_cuda']}.", '',
             f"The C++/CUDA extension compiled and loaded in {smoke['load_seconds']:.2f} seconds on the recorded first call. All {len(smoke['checks'])} add-scalar smoke cases passed.", '',
             'The L4 test suite passed 51 tests, including shape/dtype handling, rejected inputs, non-default stream ordering and CUDA Graph replay with updated inputs.', '',
             f"The separate RMSNorm sweep passed {len(numerical['cases'])} cases. It covers FP32, FP16 and BF16; widths 1–65536; 1–2048 rows; and input scales 0, 1e-7, 1 and 1e4.", '',
             '| Dtype | Cases | Maximum absolute error | Maximum relative error, denominator floor 1e-3 |',
             '|---|---:|---:|---:|']
    for dtype in ['torch.float32', 'torch.float16', 'torch.bfloat16']:
        cases = [case for case in numerical['cases'] if case['dtype'] == dtype]
        lines.append(f"| {dtype.removeprefix('torch.')} | {len(cases)} | {max(case['max_abs_error'] for case in cases):.6g} | {max(case['max_rel_error_floor_1e_3'] for case in cases):.6g} |")
    lines += ['', 'Correctness uses fixed elementwise tolerances stored per case. Aggregate maximum relative error alone is not the pass condition. Performance tuning and GEMM are subsequent stages.']
    (root / 'validation.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    report()
