"""Record static barrier/shuffle counts and compiled resource usage."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

binary = Path('.build/gpu_ops_cuda.so')
text = subprocess.check_output(['cuobjdump', '--dump-sass', str(binary)], text=True)
resources = subprocess.check_output(['cuobjdump', '--dump-resource-usage', str(binary)], text=True)
parts = re.split(r'Function\s*:\s*([^\n]+)\n', text)
records = []
for i in range(1, len(parts), 2):
    name, body = parts[i].strip(), parts[i + 1]
    if 'rmsnorm' in name:
        records.append({'function': name,
                        'static_bar_sync_instructions': len(re.findall(r'\bBAR\.SYNC\b', body)),
                        'static_shuffle_instructions': len(re.findall(r'\bSHFL\.', body))})
payload = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
           'interpretation': 'Static compiled instructions, not runtime stall counters or dynamic execution counts',
           'kernels': records}
Path('results/rmsnorm-binary-evidence.json').write_text(json.dumps(payload, indent=2) + '\n')
Path('results/rmsnorm-binary-resources.txt').write_text(resources)
print(json.dumps(payload, indent=2))
