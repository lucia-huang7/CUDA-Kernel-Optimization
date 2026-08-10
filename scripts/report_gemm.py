"""Verify GEMM source snapshots and generate the English L4 experiment report."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

root=Path(__file__).resolve().parents[1];results=root/'results'

def read(name):return json.loads((results/name).read_text())

def verify(payload,revision=None):
    for name,digest in payload['source_hashes'].items():
        data=subprocess.check_output(['git','show',revision+':'+name],cwd=root) if revision else (root/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest,name

snapshots=[('gemm-baseline.json','be4352c'),('gemm-tuning.json','abbd252'),
           ('gemm-vector-tuning.json','cb24660'),('gemm-split-tuning.json','cc0fa3e')]
for name,revision in snapshots:verify(read(name),revision)
primary=read('gemm-confirmation.json');heldout=read('gemm-heldout.json');numerical=read('gemm-correctness.json')
for payload in [primary,heldout,numerical]:verify(payload)
for payload in [primary,heldout]:
    assert payload['precision']['allow_tf32'] is False
    assert payload['precision']['pytorch_matmul_precision']=='highest'
    assert payload['iterations']==20 and payload['repeats']==30
    for case in payload['cases']:
        for entry in case['methods'].values():assert len(entry['graph_us_samples'])==30 and len(entry['wall_us_samples'])==3
assert len(numerical['cases'])==768 and all(c['passed'] for c in numerical['cases'])
assert '1330 passed' in (results/'tests-gemm-final.txt').read_text()
for tool in ['memcheck','synccheck','racecheck']:
    text=(results/f'gemm-{tool}.txt').read_text()
    assert (results/f'gemm-{tool}-exit.txt').read_text().strip()=='0'
    assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in text
    assert ('1330 passed' if tool=='memcheck' else 'PASS: 192 GEMM sanitizer cases') in text
provenance=read('gemm-final-provenance.json');verify(provenance)
profile=read('gemm-final-profile.json')
assert profile['source_sha256']==hashlib.sha256((root/'csrc/gemm.cu').read_bytes()).hexdigest()
keys=['naive','tiled','tuned','pytorch','cublas']

def table(cases,metric='graph_us_median'):
    rows=['| M | N | K | CUDA baseline | Shared tile | Tuned CUDA | PyTorch | Direct cuBLAS |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        values=' | '.join(f"{c['methods'][key][metric]:.3f}" for key in keys)
        rows.append(f"| {c['m']} | {c['n']} | {c['k']} | {values} |")
    return rows

speed=[c['methods']['naive']['graph_us_median']/c['methods']['tuned']['graph_us_median'] for c in primary['cases']]
lines=['# FP32 GEMM: tiling, register reuse and small-M parallelism','',
       f"The CUDA GEMM implementation progresses from one output per thread to shared-memory tiles, register blocking, four-element vector access and guarded split-K. Independent L4 confirmation measures **{speed[0]:.2f}×** and **{speed[1]:.2f}×** speedups over the CUDA baseline on the two target rectangular shapes. PyTorch and direct cuBLAS remain separate precision-controlled comparisons.",'',
       '## Contract and numerical controls','',
       '`C = A @ B`, with A[M,K], B[K,N] and C[M,N] in contiguous row-major FP32 storage. The operators are forward-only, use the current CUDA stream, reject autograd inputs, and handle empty output dimensions and K=0. Misaligned contiguous storage is accepted through scalar fallback paths. Dimensions must fit the integer API and the supported row grid; M is capped at 65535×16.','',
       'The custom kernels explicitly use FP32 fmaf accumulation. Existing normalization compiler options are unchanged. PyTorch uses highest float32 matmul precision with TF32 disabled. The direct cuBLAS wrapper uses cublasGemmEx with CUBLAS_COMPUTE_32F_PEDANTIC, alpha=1 and beta=0, reversing A/B dimensions for column-major cuBLAS. Pointer mode is restored after the call. This experiment measures FP32 SIMT optimization; it does not claim Tensor Core or low-precision GEMM performance.','',
       '## Locate and change','',
       'The initial baseline passed 860 full-repository tests. At M16/N4096/K4096, the first run measured 614.834 μs versus direct cuBLAS 284.192 μs. At M4096/N4096/K32, it measured 1007.178 μs versus 270.623 μs.','',
       '| Stage | Reuse or parallelism change |',
       '|---|---|',
       '| Naive | One thread computes one output; global loads inside the K loop |',
       '| Shared tile | Cooperatively load 16×16 tiles of A and B; two barriers per K tile |',
       '| Register tile | Retain 8 or 16 output accumulators per thread and reuse A/B fragments |',
       '| Vector tile | Four-element global/shared loads and output stores; alignment guards |',
       '| Small-M split-K | Compute four K partitions in parallel, then reduce their partial outputs |','',
       'The register candidates vary BM/BN/BK/TM/TN: 64/64/16/4/4, 32/64/16/2/4, 16/64/32/2/4, 16/128/32/2/8, 32/128/32/4/4 and 64/128/32/4/4. The source example tile is one candidate; its performance is remeasured on L4. Three vector layouts and split counts 4/8 are compared separately.','',
       'Register blocking alone regressed on the small-M target. Vector access improved the small-K path, while split-K improved the long-K, low-row-count path. For the latter, the 16×64 tile exposes 64 blocks; four splits expose 256 compute blocks plus a reduction kernel. The four-part workspace is 1 MiB at the target small-M shape.','',
       'The final small-M profile identifies a SIMT SGEMM kernel plus splitKreduce_kernel in both PyTorch and direct cuBLAS. Tuned CUDA likewise uses its vectorized partial-product kernel and a separate reduction. These are whole-operation comparisons, including both kernels.','',
       '### Selected dispatch','',
       '- M≤64 and K≥256: split-K with four partitions, subject to vector alignment/divisibility guards.',
       '- M≥128 and K≤64: 32×128 vector tile.',
       '- Other aligned shapes: 16×64 vector tile.',
       '- Unsupported vector alignment or N/K divisibility: general scalar register tile.',
       '', 'The guards preserve correctness; timing for an unaligned-width fallback is included below. Selection uses the tuning run only; the held-out shapes do not change these rules.','',
       '## Independent confirmation','',
       'FP32 GPU replay medians in μs. Twenty calls per capture, 30 randomized interleaved repetitions, 20 warm-up calls and five warm-up replays. First calls, including setup, are recorded separately. Inputs are fixed across repeats, so the measurements are not cold-memory or streaming-bandwidth tests.','',
       '![GEMM performance](../assets/figures/gemm_v1.png)','']+table(primary['cases'])
lines+=['','Effective arithmetic throughput uses 2MNK / elapsed time; it is not a hardware utilization counter.','',
        '| Shape M×N×K | Tuned TFLOP/s | Direct cuBLAS TFLOP/s | Baseline / tuned |',
        '|---|---:|---:|---:|']
for c,ratio in zip(primary['cases'],speed):
    lines.append(f"| {c['m']}×{c['n']}×{c['k']} | {c['methods']['tuned']['tflops']:.3f} | {c['methods']['cublas']['tflops']:.3f} | {ratio:.2f}× |")
lines+=['','### Sample spread','',
        'Tuned and cuBLAS interquartile ranges from the same 30-repeat runs, in μs. Small differences should be assessed with this spread and the held-out results.','',
        '| Shape | Tuned median [Q1,Q3] | cuBLAS median [Q1,Q3] |','|---|---:|---:|']
for c in primary['cases']:
    values=[]
    for name in ['tuned','cublas']:
        e=c['methods'][name];q=statistics.quantiles(e['graph_us_samples'],n=4,method='inclusive')
        values.append(f"{e['graph_us_median']:.3f} [{q[0]:.3f}, {q[2]:.3f}]")
    lines.append(f"| {c['m']}×{c['n']}×{c['k']} | {' | '.join(values)} |")
lines+=['','## Held-out shapes','',
        'These five shapes were measured after candidate selection with no further tuning. GPU replay medians in μs.','']+table(heldout['cases'])
lines+=['','The tuned entry improves on the naive baseline on these shapes, while cuBLAS remains faster on multiple held-out shapes and the square control. The rectangular target results are not a general replacement claim.','',
        '## Ordinary Python calls','',
        'Synchronized wall-time medians in μs, three interleaved loops of 20 calls, including validation, dispatch, workspace allocation and output allocation.','']+table(primary['cases'],'wall_us_median')
lines+=['','## Candidate evidence','',
        'All candidate measurements remain in the raw tuning files, including slower configurations. These stages have distinct source hashes and Git snapshots; do not combine their medians into a paired speedup.','',
        '| Dataset | Source revision | Candidate stage |','|---|---|---|']
for name,revision in snapshots:
    lines.append(f'| [{name}]({name}) | {revision} | '+{'gemm-baseline.json':'Naive / PyTorch / cuBLAS','gemm-tuning.json':'Shared and six register layouts','gemm-vector-tuning.json':'Three vector layouts','gemm-split-tuning.json':'Split count 4 / 8'}[name]+' |')
lines+=['','## Memory model and profiling','',
        'Unique input/output storage gives a lower-bound traffic model of 4(MK+KN+MN) bytes and arithmetic intensity 2MNK / bytes. It assumes one read of each input and one output write; shared-tile rereads, split workspace, cache behavior and write traffic can change actual transfers.','',
        '| M×N×K | Operations | Unique I/O bytes | Intensity (FLOP/byte) |','|---|---:|---:|---:|']
for c in primary['cases']:
    m,n,k=c['m'],c['n'],c['k'];flops=2*m*n*k;traffic=4*(m*k+k*n+m*n)
    lines.append(f'| {m}×{n}×{k} | {flops:,} | {traffic:,} | {flops/traffic:.2f} |')
lines+=['',f"The GPU has {provenance['gpu_sm_count']} SMs. Launch topology, static SASS and compiler resources are recorded in gemm-final-profile.json and its resource dump. NCU counter access again failed with ERR_NVGPUCTRPERM; no achieved occupancy/bandwidth or bank-conflict reduction is claimed from static evidence.",'',
        '### Compiled resources','',
       '| Kernel/layout | Registers per thread | Shared bytes/block | Local bytes |',
       '|---|---:|---:|---:|',
       '| Naive | 40 | 0 | 0 |',
       '| Shared 16×16 | 38 | 2048 | 0 |',
       '| Vector 16×64 | 56 | 10304 | 0 |',
       '| Vector 32×128 | 72 | 20608 | 0 |',
       '| Small-M split-K compute | 64 | 10304 | 0 |',
       '| Split reduction | 36 | 0 | 0 |','',
       'These values are compiler resource requirements, not achieved occupancy. The recorded custom kernel disassembly has no local-memory instruction sites.','',
       '## Correctness and tooling','',
        '| Check | Result |','|---|---:|',
        '| Full repository tests | 1330 passed |',
        '| GEMM-specific tests | 539 passed |',
        '| Numerical cases against CPU FP64 oracle | 768 passed |',
        '| Memcheck, all API errors enabled | 1330 tests; 0 errors |',
        '| Synccheck, all candidate boundary workload | 192 cases; 0 errors |',
        '| Racecheck, all candidate boundary workload | 192 cases; 0 hazards/errors/warnings |','',
        'The numerical matrix covers eight shapes, six input modes and 16 entries including PyTorch, direct cuBLAS, every candidate and tuned dispatch. Modes include zeros, constants, tiny/large magnitudes and cancellation. Fixed tolerances are rtol5e-5/atol5e-5 for K≤512 and rtol2e-4/atol2e-3 for longer K; the large-magnitude case scales absolute tolerance by the squared input scale. Every error and tolerance is retained in gemm-correctness.json.','',
        'Tests cover row/column/K tails, empty dimensions, storage offsets, invalid contracts/configurations, non-default streams and CUDA Graph replay after changing inputs. The compact synchronization/race workload exercises all 16 entries on tail/alignment cases and empty dimensions. The independent oracle runs FP64 arithmetic on the CPU; no sanitizer name filters or suppressions are used.','',
        '## Sources','',
        '- [CUDA shared-memory GEMM guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)',
        '- [cuBLAS precision modes and GEMM API](https://docs.nvidia.com/cuda/cublas/index.html)',
        '- [PyTorch CUDA numerical precision](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)']
(results/'gemm.md').write_text('\n'.join(lines)+'\n')
print('Verified GEMM source snapshots, timing protocol and sanitizer evidence.')
print('Naive-to-tuned speedups:',[round(v,3) for v in speed])
