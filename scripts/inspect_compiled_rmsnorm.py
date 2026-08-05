"""Inspect generated Inductor RMSNorm kernels without executing cached code."""
import hashlib
import json
from pathlib import Path

cache = Path('/tmp/torchinductor_root')
records = []
for path in sorted(cache.rglob('*.py')):
    text = path.read_text(errors='replace')
    if '1536' not in text or 'rsqrt' not in text or 'triton_heuristics' not in text:
        continue
    relevant = [line for line in text.splitlines() if any(key in line for key in
                ['persistent_reduction', 'size_hints', 'num_warps', 'XBLOCK', 'RBLOCK',
                 'tl.load', 'tl.store', 'tl.sum', 'tl.rsqrt', 'rnumel', 'xnumel',
                 'def triton_', '.to(tl.bfloat16)', 'triton_meta'])]
    records.append({'file': path.name, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'excerpt': relevant})
payload = {'interpretation': 'Generated-code inspection only; load sites and reduction layout are static evidence.',
           'kernels': records}
Path('results/compiled-rmsnorm-inspection.json').write_text(json.dumps(payload, indent=2) + '\n')
for item in records:
    print(item['file'])
    print('\n'.join(item['excerpt'])[:6000])
print('MATCHING_FILES', len(records))
