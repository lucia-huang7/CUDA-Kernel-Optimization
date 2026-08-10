"""Save FP32 GEMM error matrices against a CPU FP64 oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import gpu_ops
from reference.gemm import configure_precision,gemm_oracle,tolerance
from scripts.benchmark_gemm import methods


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();output=Path(args.output)
    if output.exists():raise SystemExit('Choose a fresh output path.')
    gpu_ops.initialize();configure_precision();torch.set_num_threads(4);torch.manual_seed(20261016)
    shapes=[(1,1,1),(3,5,7),(17,65,33),(65,129,35),(16,257,1025),
            (16,4096,4096),(4096,4096,32),(512,512,512)]
    records=[]
    for m,n,k in shapes:
        for mode in ['standard','zero','constant','tiny','large','cancellation']:
            a=torch.randn(m,k,device='cuda');b=torch.randn(k,n,device='cuda');scale=1.0
            if mode=='zero':a.zero_()
            elif mode=='constant':a.fill_(1);b.fill_(2)
            elif mode=='tiny':scale=1e-4;a*=scale;b*=scale
            elif mode=='large':scale=100.;a*=scale;b*=scale
            elif mode=='cancellation':a.fill_(1);b[::2]=1;b[1::2]=-1
            expected=gemm_oracle(a,b);tol=tolerance(k).copy();tol['atol']*=max(1.0,scale*scale)
            for name,fn in methods().items():
                actual=fn(a,b);torch.testing.assert_close(actual,expected,**tol)
                error=(actual-expected).abs()
                records.append({'variant':name,'m':m,'n':n,'k':k,'mode':mode,'tolerance':tol,
                                'max_abs_error':error.max().item(),
                                'max_rel_error_floor_1e_3':(error/expected.abs().clamp_min(1e-3)).max().item(),
                                'relative_l2_error':(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected).clamp_min(1e-30)).item(),
                                'passed':True})
            print('Validated',m,n,k,mode,flush=True)
    files=['csrc/gemm.cu','csrc/bindings.cpp','gpu_ops/extension.py','gpu_ops/ops.py','reference/gemm.py',
           'scripts/validate_gemm.py','scripts/benchmark_gemm.py']
    payload={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'seed':20261016,
             'oracle':'CPU FP64 matmul, final FP32 rounding','cpu_threads':4,'allow_tf32':False,
             'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files},'cases':records}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+'\n')
    print('PASS:',len(records),'GEMM numerical cases.')


if __name__=='__main__':main()
