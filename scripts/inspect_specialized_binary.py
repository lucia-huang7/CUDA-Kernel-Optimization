"""Record static vector instructions, barriers and resources for RMSNorm variants."""
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
    if 'rmsnorm' not in name:
        continue
    instructions = re.findall(r'/\*\w+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)', body)
    records.append({'function': name,
                    'barriers': sum(op.startswith('BAR.SYNC') for op in instructions),
                    'global_load_instructions': [op for op in instructions if op.startswith('LDG')],
                    'global_store_instructions': [op for op in instructions if op.startswith('STG')],
                    'load_128bit_sites': sum(op.startswith('LDG') and '.128' in op for op in instructions),
                    'store_128bit_sites': sum(op.startswith('STG') and '.128' in op for op in instructions),
                    'local_memory_instruction_sites': sum(op.startswith(('LDL', 'STL')) for op in instructions)})
payload = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
           'interpretation': 'Static SASS instruction sites; no dynamic stalls, traffic or utilization measured.',
           'kernels': records}
Path('results/rmsnorm-specialized-binary.json').write_text(json.dumps(payload, indent=2) + '\n')
Path('results/rmsnorm-specialized-resources.txt').write_text(resources)
print('Static RMSNorm kernel records:', len(records))
for k in records:
    if 'BFloat16' in k['function']:
        print(k['function'], 'BAR', k['barriers'], 'LD128', k['load_128bit_sites'],
              'ST128', k['store_128bit_sites'], 'LOCAL', k['local_memory_instruction_sites'])
