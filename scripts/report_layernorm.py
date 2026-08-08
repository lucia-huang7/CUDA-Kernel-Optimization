"""Verify saved LayerNorm experiments and generate the English report."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parents[1];results=root/'results'

def read(name):return json.loads((results/name).read_text())

def verify_sources(payload,revision=None):
    for name,digest in payload['source_hashes'].items():
        data=subprocess.check_output(['git','show',revision+':'+name],cwd=root) if revision else (root/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest,name

baseline=read('layernorm-baseline.json');verify_sources(baseline,'01e1068')
primary=read('layernorm-confirmation.json');tuning=read('layernorm-tuning.json');heldout=read('layernorm-heldout.json')
extra={name:read(name) for name in ['layernorm-float16.json','layernorm-float32.json',
                                    'layernorm-width-768.json','layernorm-width-1537.json','layernorm-width-4096.json']}
for payload in [primary,tuning,heldout,*extra.values()]:
    verify_sources(payload)
    assert payload['iterations']==100 and payload['repeats']==30
    for case in payload['cases']:
        for entry in case['methods'].values():assert len(entry['graph_us_samples'])==30 and len(entry['wall_us_samples'])==3
numerical=read('layernorm-final-correctness.json');verify_sources(numerical)
assert len(numerical['cases'])==720 and all(c['passed'] for c in numerical['cases'])
assert '791 passed' in (results/'tests-layernorm-optimized.txt').read_text()
for tool,count in [('memcheck',791),('synccheck',311),('racecheck',311)]:
    log=(results/f'layernorm-final-{tool}.txt').read_text()
    assert f'{count} passed' in log
    assert (results/f'layernorm-final-{tool}-exit.txt').read_text().strip()=='0'
    assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in log
profile=read('layernorm-optimized-profile.json')
assert profile['source_sha256']==hashlib.sha256((root/'csrc/layernorm.cu').read_bytes()).hexdigest()
keys=['cuda_shared','cuda_vector','pytorch_native','pytorch_compiled']

def table(cases,wall=False):
    metric='wall_us_median' if wall else 'graph_us_median'
    rows=['| Rows | Width | Dtype | CUDA baseline | Vector entry | PyTorch native | PyTorch compiled |',
          '|---:|---:|---|---:|---:|---:|---:|']
    for c in cases:
        values=' | '.join(f"{c['methods'][k][metric]:.3f}" for k in keys)
        rows.append(f"| {c['rows']} | {c['width']} | {c['dtype']} | {values} |")
    return rows

reduction=[100*(1-c['methods']['cuda_vector']['graph_us_median']/c['methods']['cuda_shared']['graph_us_median']) for c in primary['cases']]
native=[c['methods']['pytorch_native']['graph_us_median']/c['methods']['cuda_vector']['graph_us_median'] for c in primary['cases']]
last=primary['cases'][-1]['methods']
lines=['# LayerNorm: CUDA implementation and L4 optimization','',
       f"The vectorized width-1536 kernel reduces GPU replay latency by **{min(reduction):.1f}%–{max(reduction):.1f}%** versus the shared-tree CUDA baseline on six independent confirmation shapes. It is **{min(native):.2f}×–{max(native):.2f}×** faster than native PyTorch on these BF16 shapes. At 2048 rows it takes **{last['cuda_vector']['graph_us_median']:.3f} μs** versus **{last['pytorch_native']['graph_us_median']:.3f} μs** for native PyTorch and **{last['pytorch_compiled']['graph_us_median']:.3f} μs** for compiled reference.",'',
       '## Semantics and implementation','',
       '`y = ((x - mean(x)) / sqrt(mean((x - mean(x))²) + eps)) * weight + bias`, with population variance, FP32 statistics/affine arithmetic and a final cast to the input dtype. Default epsilon is 1e-5. Unlike the RMSNorm path, normalization is not rounded to BF16/FP16 before affine scaling.','',
       'The forward-only interface accepts contiguous CUDA tensors of shape [rows, width], width 1–65536, with contiguous same-device and same-dtype weight/bias vectors. FP32, FP16 and BF16 are supported. Empty rows return an empty output. Autograd inputs and invalid contracts are rejected.','',
       'Statistics are centered around the first row element: reduce x-anchor, then reduce squared deviations in that shifted coordinate system. This avoids subtracting two large raw moments and preserves small variations on an offset input. An independent CPU FP64 oracle checks the mathematical result.','',
       '## Locate, change, measure','',
       'The initial implementation passed 560 combined tests. Its first BF16 run measured 47.695 μs at 2048 rows, versus native PyTorch 23.539 μs. The profiler observed 8 CUDA launches per decomposed reference call, 1 for native PyTorch and 1 for the custom kernel. Native PyTorch was already fused.','',
       '| Variant | Mechanism | Width-1536 threads/row | Static block barrier sites |',
       '|---|---|---:|---:|',
       '| Shared baseline | Shared-tree mean and variance; input reread | 256 | 20 |',
       '| Warp | Warp shuffles and two separate shared scratch arrays | 128 | 4 |',
       '| Cached | Fixed-width unrolling; retain 12 FP32 values/thread | 128 | 4 |',
       '| Vector | Cached values plus four-element input/affine/output packs | 128 | 4 |','',
       'All four CUDA variants retain the same anchored centered-variance formula. Separate mean/variance scratch arrays prevent reuse before all warps finish reading. Vector packs require 8-byte alignment for FP16/BF16 and 16-byte alignment for FP32; any misaligned input, weight or bias selects the scalar cached path. Other widths use the general warp path.','',
       '### Candidate ablation','',
       'GPU replay medians in μs, from one interleaved tuning run. Failed or weaker candidates remain in the table.','',
       '| Rows | Shared | Warp | Cached | Vector | Native | Compiled |','|---:|---:|---:|---:|---:|---:|---:|']
for c in tuning['cases']:
    vals=' | '.join(f"{c['methods'][k]['graph_us_median']:.3f}" for k in ['cuda_shared','cuda_warp','cuda_cached','cuda_vector','pytorch_native','pytorch_compiled'])
    lines.append(f"| {c['rows']} | {vals} |")
lines+=['','Warp reduction alone regressed on small inputs. Register retention helped, and vector packs produced the strongest overall candidate. These are controlled empirical comparisons, not hardware stall-counter measurements.','',
        '## Independent confirmation','',
        'NVIDIA L4, PyTorch 2.8.0+cu128, CUDA 12.8, BF16, width 1536. Each median uses 30 randomized interleaved repetitions, 100 calls per CUDA Graph replay, 20 warm-up calls and 5 warm-up replays. First-call compilation/loading is saved separately. Fixed buffers are reused; these are warm-cache microbenchmarks.','',
        '![LayerNorm latency](../assets/figures/layernorm_v1.png)','']+table(primary['cases'])
lines+=['','A fresh process confirmed the six tuning shapes. Five additional row counts were then measured without further kernel changes:','']+table(heldout['cases'])
lines+=['','Compiled reference remains faster on the smallest row counts; vectorization is not a universal win over every framework backend.','',
        '## Precision and width checks','',
        'The same four-method protocol was repeated for FP16/FP32 and additional BF16 widths. At widths other than 1536, the vector entry dispatches to the general warp kernel; their performance is shown explicitly. GPU replay medians in μs.','']
for payload in extra.values():lines+=table(payload['cases'])+['']
lines+=['## Ordinary Python calls','',
        'Synchronized wall-time medians in μs from three interleaved loops of 100 calls, including Python dispatch, validation and output allocation. They are separate from graph replay latency.','']+table(primary['cases'],wall=True)
lines+=['','## Correctness and tooling','',
        '| Check | Result |','|---|---:|',
        '| Full repository tests, including RMSNorm regression | 791 passed |',
        '| LayerNorm tests across four variants | 311 passed |',
        '| Independent CPU FP64-oracle numerical matrix | 720 passed |',
        '| Memcheck, all API errors enabled, full suite | 791 tests; 0 errors |',
        '| Synccheck, LayerNorm suite | 311 tests; 0 errors |',
        '| Racecheck, LayerNorm suite | 311 tests; 0 hazards/errors/warnings |','',
        'The numerical matrix uses 10 shapes, three dtypes, four variants and six input modes: standard, zero, constant, tiny, large and shifted. Tolerances were fixed before tuning: FP32 rtol2e-5/atol5e-6; FP16 rtol/atol2e-3; BF16 rtol/atol2e-2. Tests cover tail widths, offset-but-contiguous storage, invalid contracts, non-default stream ordering and graph replay with changed inputs.','',
        'Explicit extension startup loading preserves the previously validated sanitizer initialization order. No sanitizer errors are suppressed. The NCU attempt again returned ERR_NVGPUCTRPERM; no host-driver permissions were changed. Static SASS confirms 20 versus 4 barrier sites and no local-memory instruction sites in the compiled LayerNorm kernels. Register/shared-memory counts are preserved in the resource dump; achieved occupancy/bandwidth is not inferred from them.','',
        '### Reference-path racecheck isolation','',
        'The first full LayerNorm racecheck run returned 24 analysis reports, all naming PyTorch FP64 MeanOps reduce_kernel, with no custom LayerNorm kernel in those reports. The independent oracle had been running FP64 means on the GPU. Its failure log remains in layernorm-racecheck.txt.','',
        'A PyTorch-only reproducer in diagnostics/torch_fp64_mean.py separates this path from the extension. The final oracle executes FP64 statistics and affine arithmetic on the CPU, then transfers the expected result for comparison. The same GPU operator cases remain checked with no name filters, exclusions or suppressions. CUDA kernel sources and all timing datasets are unchanged by the oracle change. The original and newer-tool control logs are retained; no vendor-internal cause is asserted.','',
        '## Evidence','',
        '- `layernorm-baseline.json` and `layernorm-baseline-profile.*`: initial source snapshot at commit 01e1068.',
        '- `layernorm-tuning.json`, `layernorm-confirmation.json`, `layernorm-heldout.json`: ablation and independent confirmation.',
        '- `layernorm-float16.json`, `layernorm-float32.json`, `layernorm-width-*.json`: dtype and fallback measurements.',
        '- `layernorm-final-correctness.json`, `tests-layernorm-optimized.txt`, `layernorm-final-memcheck.txt`, `layernorm-final-synccheck.txt`, `layernorm-final-racecheck.txt`: numerical and execution checks.',
        '- `layernorm-optimized-profile.json`, `layernorm-optimized-profile.resources.txt`, `layernorm-ncu.txt`: launch topology, static instruction/resource evidence and counter-access result.','',
        '## Conclusions','',
        'The width-1536 vectorized layout converts the reduction and memory-access changes into measured kernel latency gains. Small inputs still expose fixed launch/reduction costs, while non-specialized widths can favor native or compiled PyTorch. Full-model integration and GEMM are separate stages.','',
        '## References','',
        '- [PyTorch 2.8 LayerNorm semantics](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html)',
        '- [PyTorch 2.8 native CUDA LayerNorm implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu)',
        '- [CUDA 12.8 memory alignment](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#device-memory-accesses)']
(results/'layernorm.md').write_text('\n'.join(lines)+'\n')
print(f'Validated LayerNorm evidence: latency reduction {min(reduction):.1f}%–{max(reduction):.1f}%; native speedup {min(native):.2f}x–{max(native):.2f}x.')
