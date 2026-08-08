"""Validate LayerNorm against an independent FP64 oracle and save all errors."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from gpu_ops import initialize, layernorm, layernorm_warp, layernorm_cached, layernorm_vector

TOLERANCES={torch.float32:dict(rtol=2e-5,atol=5e-6),torch.float16:dict(rtol=2e-3,atol=2e-3),
            torch.bfloat16:dict(rtol=2e-2,atol=2e-2)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise SystemExit('Choose a fresh output path.')
    initialize();torch.manual_seed(20261016);records=[]
    methods={'shared':layernorm,'warp':layernorm_warp,'cached':layernorm_cached,'vector':layernorm_vector}
    shapes=[(1,1),(3,31),(7,33),(8,256),(7,768),(5,1536),(9,1537),(2,4096),(2,65536),(2048,1536)]
    for dtype,tolerance in TOLERANCES.items():
        for rows,width in shapes:
            for mode in ['standard','zero','constant','tiny','large','shifted']:
                x=torch.randn(rows,width,device='cuda',dtype=dtype)
                if mode=='zero':x.zero_()
                elif mode=='constant':x.fill_(10000)
                elif mode=='tiny':x*=1e-7
                elif mode=='large':x*=1e4
                elif mode=='shifted':x=x*0.01+10
                w=torch.randn(width,device='cuda',dtype=dtype);b=torch.randn_like(w)
                values=x.cpu().double();centered=values-values.mean(-1,keepdim=True)
                expected=(centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)*w.cpu().double()+b.cpu().double()).to(dtype=dtype,device=x.device)
                for name,fn in methods.items():
                    actual=fn(x,w,b);torch.testing.assert_close(actual,expected,**tolerance)
                    errors=(actual.float()-expected.float()).abs()
                    records.append({'variant':name,'rows':rows,'width':width,'dtype':str(dtype),'mode':mode,
                                    'max_abs_error':errors.max().item(),
                                    'max_rel_error_floor_1e_3':(errors/expected.float().abs().clamp_min(1e-3)).max().item(),
                                    'tolerance':tolerance,'passed':True})
    files=['csrc/layernorm.cu','csrc/bindings.cpp','gpu_ops/extension.py','gpu_ops/ops.py','scripts/validate_layernorm.py']
    payload={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'eps':1e-5,'seed':20261016,
             'oracle':'CPU FP64 centered population variance and affine; final dtype cast only',
             'source_hashes':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},'cases':records}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS:',len(records),'LayerNorm numerical cases across FP32, FP16 and BF16.')


if __name__=='__main__':main()
