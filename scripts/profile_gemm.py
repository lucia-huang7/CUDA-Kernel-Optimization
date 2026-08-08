"""Record GEMM launch topology, static SASS and compiled resource requirements."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import torch
import gpu_ops
from reference.gemm import configure_precision
from scripts.benchmark_gemm import methods


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--shape',default='16x4096x4096');args=p.parse_args()
    prefix=Path(args.prefix)
    if prefix.with_suffix('.json').exists():raise SystemExit('Choose a fresh prefix.')
    gpu_ops.initialize();configure_precision();m,n,k=map(int,args.shape.split('x'))
    a=torch.randn(m,k,device='cuda');b=torch.randn(k,n,device='cuda');topology={}
    with torch.inference_mode():
        for name,fn in methods().items():
            for _ in range(5):fn(a,b)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):fn(a,b)
                torch.cuda.synchronize()
            events=[{'name':e.name,'device_us':e.device_time_total} for e in prof.events()
                    if e.device_type==torch.autograd.DeviceType.CUDA]
            topology[name]={'calls':3,'events':events,'cuda_events_per_call':len(events)/3}
    binary=Path('.build/gpu_ops_cuda.so')
    text=subprocess.check_output(['cuobjdump','--dump-sass',str(binary)],text=True)
    resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(binary)],text=True)
    chunks=re.split(r'Function\s*:\s*([^\n]+)\n',text);kernels=[]
    for i in range(1,len(chunks),2):
        name,body=chunks[i].strip(),chunks[i+1]
        if 'gemm_' not in name:continue
        ops=re.findall(r'/\*\w+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)
        kernels.append({'name':name,'barrier_sites':sum(o.startswith('BAR.SYNC') for o in ops),
                        'ffma_sites':sum(o.startswith('FFMA') for o in ops),
                        'shared_instruction_sites':sum(o.startswith(('LDS','STS')) for o in ops),
                        'local_memory_sites':sum(o.startswith(('LDL','STL')) for o in ops)})
    payload={'shape':[m,n,k],'topology':topology,'static_kernels':kernels,
             'source_sha256':hashlib.sha256(Path('csrc/gemm.cu').read_bytes()).hexdigest(),
             'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
             'interpretation':'Launch topology and static instruction sites, not measured bandwidth or stalls'}
    prefix.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n');prefix.with_suffix('.resources.txt').write_text(resource)
    for name,entry in topology.items():print(name,entry['cuda_events_per_call'],'CUDA events/call')
    for kernel in kernels:print(kernel)


if __name__=='__main__':main()
