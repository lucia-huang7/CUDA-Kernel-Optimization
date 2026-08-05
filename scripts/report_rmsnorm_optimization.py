"""Generate an optimization report from saved timing and validation records."""
import json
import statistics
from pathlib import Path

root = Path('results')
performance = json.loads((root / 'rmsnorm-warp-comparison.json').read_text())
correctness = json.loads((root / 'rmsnorm-warp-correctness.json').read_text())
binary = json.loads((root / 'rmsnorm-binary-evidence.json').read_text())
assert all(case['passed'] for case in correctness['cases'])
assert '98 passed' in (root / 'tests-warp.txt').read_text()
for case in performance['cases']:
    for entry in case['methods'].values():
        assert entry['graph_us_median'] == statistics.median(entry['graph_us_samples'])
        assert entry['wall_us_median'] == statistics.median(entry['wall_us_samples'])
lines = ['# RMSNorm Warp Reduction Experiment', '',
         'NVIDIA L4; BF16; width 1536; one 256-thread block per row. The CUDA variants differ only in their block reduction: a shared-memory tree versus warp shuffles and eight shared partial sums.', '',
         '## GPU replay latency', '',
         'The table reports median CUDA event time per invocation in a 100-call CUDA Graph sequence, with 30 interleaved repeats. Compilation, Python dispatch and output-allocation setup are outside replay timing. Inputs and weights are reused; this is a warm-cache microbenchmark.', '',
         '| Rows | PyTorch eager (us) | CUDA shared (us) | CUDA warp (us) | PyTorch compiled (us) | Shared / warp |',
         '|---:|---:|---:|---:|---:|---:|']
for case in performance['cases']:
    m = case['methods']
    latency = lambda name: m[name]['graph_us_median']
    lines.append(f"| {case['rows']} | {latency('pytorch_eager'):.3f} | {latency('cuda_shared'):.3f} | {latency('cuda_warp'):.3f} | {latency('pytorch_compiled'):.3f} | {latency('cuda_shared') / latency('cuda_warp'):.2f}x |")
lines += ['', '## Ordinary Python API wall time', '',
          'Each value is the median of three synchronized 100-call loops, including Python validation/dispatch and output allocation. These are operator API calls, not complete model or HTTP request timings.', '',
          '| Rows | CUDA shared (us) | CUDA warp (us) | Shared / warp |', '|---:|---:|---:|---:|']
for case in performance['cases']:
    m = case['methods']
    a, b = m['cuda_shared']['wall_us_median'], m['cuda_warp']['wall_us_median']
    lines.append(f"| {case['rows']} | {a:.3f} | {b:.3f} | {a / b:.2f}x |")
lines += ['', '## Mechanism and correctness', '',
          'Baseline source executes nine block-wide barriers: one initial publication plus eight shared-tree stages. Warp reduction uses five shuffle steps per warp, publishes eight warp sums, reduces them in the first warp, then publishes the total with two block-wide barriers. Input loads, row mapping, block size, output allocation and precision semantics remain unchanged.', '',
          'Compiled instruction evidence is recorded below. Counts are static SASS instructions, not stall metrics or measured dynamic instruction totals.', '',
          '| Variant | Dtype | Static BAR.SYNC | Static SHFL |', '|---|---|---:|---:|']
for entry in binary['kernels']:
    name = entry['function']
    dtype = 'bfloat16' if 'BFloat16' in name else 'float16' if 'Half' in name else 'float32' if 'IfEE' in name else None
    if dtype:
        variant = 'warp' if 'rmsnorm_warp_kernel' in name else 'shared'
        lines.append(f"| {variant} | {dtype} | {entry['static_bar_sync_instructions']} | {entry['static_shuffle_instructions']} |")
lines += ['', f"All 98 tests passed. The separate sweep passed {len(correctness['cases'])} numerical cases across both variants, FP32/FP16/BF16, tails, extreme tested magnitudes and large widths. CUDA stream ordering, graph replay and contract rejection are tested for both variants.", '',
          '## Profiling access', '',
          'Nsight Compute 2025.1.1 was invoked with SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats and MemoryWorkloadAnalysis. The cloud driver denied counter access with ERR_NVGPUCTRPERM; the original output and exit status are retained. No stall, achieved bandwidth, achieved occupancy or utilization percentage is claimed. Static resource/disassembly inspection and unprofiled timing support a controlled reduction experiment; they do not establish that barriers are the only bottleneck.', '',
          '## Evidence', '',
          '- `rmsnorm-baseline.json`: timing before the warp variant was added.',
          '- `rmsnorm-warp-comparison.json`: simultaneous reference/shared/warp/compiled controls and raw samples.',
          '- `rmsnorm-warp-correctness.json`, `tests-warp.txt`: numerical and execution-semantics validation.',
          '- `rmsnorm-binary-evidence.json`, `rmsnorm-binary-resources.txt`: compiled instruction and resource evidence.',
          '- `ncu-shared-2048.csv`, `ncu-shared-exit.txt`: hardware-counter access failure.', '',
          'The compiled PyTorch control is kept separate from the eager reference. Operator results do not establish model-level speedup. LayerNorm and GEMM remain subsequent stages.']
(root / 'rmsnorm-optimization.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:18]))
