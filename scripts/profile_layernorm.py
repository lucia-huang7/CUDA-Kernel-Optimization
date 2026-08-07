"""Save per-method CUDA launch topology and static LayerNorm binary evidence."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import torch
import gpu_ops
from reference.layernorm import layernorm_native, layernorm_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);args=p.parse_args()
    prefix=Path(args.prefix)
    if prefix.with_suffix('.json').exists():raise SystemExit('Choose a fresh prefix.')
    gpu_ops.initialize()
    x=torch.randn(512,1536,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(1536,device='cuda',dtype=x.dtype);b=torch.randn_like(w)
    methods={'pytorch_eager':layernorm_reference,'pytorch_native':layernorm_native,'cuda_shared':gpu_ops.layernorm}
    for name in ['warp','cached','vector']:
        if hasattr(gpu_ops,'layernorm_'+name):methods['cuda_'+name]=getattr(gpu_ops,'layernorm_'+name)
    records={}
    with torch.inference_mode():
        for name,fn in methods.items():
            for _ in range(20):fn(x,w,b)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(10):fn(x,w,b)
                torch.cuda.synchronize()
            events=[{'name':e.name,'device_us':e.device_time_total} for e in prof.events()
                    if e.device_type==torch.autograd.DeviceType.CUDA]
            records[name]={'calls':10,'cuda_events':events,'events_per_call':len(events)/10}
    binary=Path('.build/gpu_ops_cuda.so')
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True)
    resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(binary)],text=True)
    parts=re.split(r'Function\s*:\s*([^\n]+)\n',sass);kernels=[]
    for i in range(1,len(parts),2):
        name,body=parts[i].strip(),parts[i+1]
        if 'layernorm' not in name:continue
        ops=re.findall(r'/\*\w+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)
        kernels.append({'name':name,'barrier_sites':sum(o.startswith('BAR.SYNC') for o in ops),
                        'global_load_sites':[o for o in ops if o.startswith('LDG')],
                        'global_store_sites':[o for o in ops if o.startswith('STG')],
                        'local_memory_sites':sum(o.startswith(('LDL','STL')) for o in ops)})
    payload={'shape':[512,1536],'dtype':'bfloat16','launch_topology':records,'static_kernel_records':kernels,
             'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
             'source_sha256':hashlib.sha256(Path('csrc/layernorm.cu').read_bytes()).hexdigest(),
             'interpretation':'Profiler launch topology and static instruction sites; hardware counters are separate.'}
    prefix.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n')
    prefix.with_suffix('.resources.txt').write_text(resources)
    for name,record in records.items():print(name,record['events_per_call'],'CUDA launches/call')
    for k in kernels:
        if 'BFloat16' in k['name']:print(k['name'],'BAR',k['barrier_sites'],'LOCAL',k['local_memory_sites'])


if __name__=='__main__':main()
