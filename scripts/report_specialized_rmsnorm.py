"""Report fixed-width RMSNorm ablations, confirmation and held-out row counts."""
import json
import statistics
from pathlib import Path

root = Path('results')
def read(name):
    return json.loads((root / name).read_text())

confirmation = read('rmsnorm-specialized-confirmation.json')
heldout = read('rmsnorm-specialized-heldout.json')
controls = read('rmsnorm-specialized-controls.json')
correctness = read('rmsnorm-specialized-final-correctness.json')
assert '480 passed' in (root / 'tests-specialized-final.txt').read_text()
assert len(correctness['cases']) == 924 and all(c['passed'] for c in correctness['cases'])
for dataset in [confirmation, heldout, controls]:
    for case in dataset['cases']:
        for entry in case['methods'].values():
            assert entry['graph_us_median'] == statistics.median(entry['graph_us_samples'])
            assert entry['wall_us_median'] == statistics.median(entry['wall_us_samples'])

def latency(case, method, wall=False):
    return case['methods'][method]['wall_us_median' if wall else 'graph_us_median']

lines = ['# RMSNorm Register and Vector Layout Experiment', '',
         'NVIDIA L4, BF16, width 1536. Primary results use a fresh-process confirmation run after candidate evaluation. Row counts 2, 16, 64, 256 and 1024 were measured separately after tuning.', '',
         '## Diagnosis and changes', '',
         'Generated Inductor code also contains two input load sites. That observation does not make repeated loads the sole explanation for the earlier gap. The prior CUDA warp kernel uses runtime width, a 256-thread row block and scalar loads. This experiment separates fixed-width reload, register reuse, row grouping and vectorized layouts.', '',
         '| Variant | Row mapping | Input retention and access |', '|---|---|---|',
         '| cuda_warp | 256 threads per row | Runtime-width loops, input reread |',
         '| cuda_reload | 256 threads per row | Width-1536 unrolled loops, input reread control |',
         '| cuda_cached | 256 threads per row | Width-1536 unrolled loops, six FP32 values retained per thread |',
         '| cuda_rows | One warp per row, four rows per block | 48 FP32 values retained per thread, scalar access |',
         '| cuda_vector | One warp per row, four rows per block | Retained values and aligned 16-byte packs |',
         '| cuda_vector128 | 128 threads per row | Twelve FP32 values retained per thread, four-element packs |', '',
         'The selected vector128 path uses 8-byte loads/stores for BF16/FP16 and 16-byte loads/stores for FP32. Two block-wide barriers remain. Widths other than 1536 use the previous warp implementation. Misaligned contiguous storage falls back to a scalar-safe cached path; no runtime timing-based autotuning or row-count lookup is used.', '',
         '## Confirmation run', '',
         'Each value is the median CUDA event time per invocation in a 100-call graph replay, across 30 randomized method-order repetitions. Inputs and weights are reused. Compilation, Python dispatch and allocation setup are outside replay timing.', '',
         '| Rows | Previous CUDA (us) | Cached 256 (us) | Vector 128 (us) | PyTorch compiled (us) | Latency reduction vs previous |',
         '|---:|---:|---:|---:|---:|---:|']
for case in confirmation['cases']:
    a,b,c,d = [latency(case,m) for m in ['cuda_warp','cuda_cached','cuda_vector128','pytorch_compiled']]
    lines.append(f"| {case['rows']} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {(1-c/a)*100:.1f}% |")
lines += ['', '## Held-out row counts', '',
          'These shapes were not used in the candidate tuning run. Width, dtype, graph timing and numerical tolerance remain unchanged.', '',
          '| Rows | Previous CUDA (us) | Vector 128 (us) | PyTorch compiled (us) | Vector / compiled latency |',
          '|---:|---:|---:|---:|---:|']
for case in heldout['cases']:
    a,b,c=[latency(case,m) for m in ['cuda_warp','cuda_vector128','pytorch_compiled']]
    lines.append(f"| {case['rows']} | {a:.3f} | {b:.3f} | {c:.3f} | {b/c:.2f}x |")
lines += ['', '## Ablation controls', '',
          'The following data comes from the candidate/control run, separate from the confirmation run. The scalar multi-row candidate regressed; it remains selectable and its measurements are preserved.', '',
          '| Rows | Runtime warp | Fixed reload | Cached 256 | Scalar rows | Vector rows | Vector 128 | Compiled |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
for case in controls['cases']:
    methods = ['cuda_warp','cuda_reload','cuda_cached','cuda_rows','cuda_vector','cuda_vector128','pytorch_compiled']
    lines.append('| '+str(case['rows'])+' | '+' | '.join(f'{latency(case,m):.3f}' for m in methods)+' |')
lines += ['', 'All entries are microseconds. Fixed-width specialization changes loop/address code as well as allowing register retention. Compare fixed reload with cached to isolate retention under the same width and block mapping; compare runtime warp with fixed reload for the specialization effect. The vector128 candidate changes thread mapping and vector width together, so those contributions are not independently isolated.', '',
          '## Ordinary Python API calls', '',
          'Median of three synchronized 100-call loops, including dispatch, input validation and output allocation. This is not model or HTTP latency.', '',
          '| Rows | Previous CUDA (us) | Vector 128 (us) | PyTorch compiled (us) |', '|---:|---:|---:|---:|']
for case in confirmation['cases']:
    lines.append('| '+str(case['rows'])+' | '+' | '.join(f'{latency(case,m,True):.3f}' for m in ['cuda_warp','cuda_vector128','pytorch_compiled'])+' |')
lines += ['', '## Correctness and compiled evidence', '',
          'All 480 tests passed. The separate sweep passed 924 cases across seven variants and FP32/FP16/BF16. Tests include row tails, offset-but-contiguous input/weight storage, unsupported shapes/layouts, CUDA stream ordering and graph replay. All paths preserve the normalized-activation cast before weight multiplication.', '',
          'The compiled resource/disassembly records retain vector instruction sites, barrier counts, local-memory instruction sites and register/shared-storage usage. These are static compiler evidence, not achieved bandwidth, occupancy or runtime stall measurements. The previous NCU attempt was denied with ERR_NVGPUCTRPERM; no driver permissions were changed.', '',
          'Selected BF16/FP16 SASS uses six 64-bit global-load sites and three 64-bit store sites; FP32 uses the corresponding 128-bit instructions. The selected BF16 kernel uses 28 registers per thread and 16 bytes of static shared storage, with no local-memory instruction sites. These are compiled resource observations rather than achieved occupancy measurements.', '',
          '## Evidence files', '',
          '- `rmsnorm-specialized-tuning.json`: first candidate exploration.',
          '- `rmsnorm-specialized-controls.json`: specialization/reload/cache/layout controls.',
          '- `rmsnorm-specialized-confirmation.json`: primary fresh-process results.',
          '- `rmsnorm-specialized-heldout.json`: additional row counts.',
          '- `rmsnorm-specialized-final-correctness.json`, `tests-specialized-final.txt`: numerical and execution-semantics checks.',
          '- `compiled-rmsnorm-inspection.json`: generated Inductor snippets.',
          '- `rmsnorm-specialized-binary.json`, `rmsnorm-specialized-resources.txt`: static binary evidence.', '',
          'The optimization targets width 1536. Other widths retain the tested fallback; they are not claimed faster. PyTorch compiled remains a stronger control than eager execution and can remain faster on small shapes. Full-model benefit is a separate experiment.']
if (root / 'specialized-memcheck.txt').exists():
    lines += ['', '## Sanitizer diagnostics', '',
              'The 60 alignment/tail comparisons completed, but Compute Sanitizer reported a CUDA_ERROR_INVALID_HANDLE at cuKernelGetFunction. Eager module loading did not remove the diagnostic. This is retained as an unresolved instrumentation/API result, not a clean memcheck pass.', '',
              'The original add-scalar extension reproduced the same cuKernelGetFunction diagnostic; a PyTorch-only control completed with zero errors. This narrows the issue to the extension/tool/runtime path without proving a root cause. See the separate control logs and exit files. Numerical/stream/graph validation and timing conclusions are separate from this unresolved sanitizer check.']
(root / 'rmsnorm-specialized.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[24:38]))
