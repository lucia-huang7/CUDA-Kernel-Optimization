"""Interleaved FP32 GEMM replay, wall time and precision-controlled cuBLAS baselines."""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from pathlib import Path
import torch
import gpu_ops
from reference.gemm import configure_precision,tolerance
from scripts.check_env import collect_environment
from scripts.benchmark_rmsnorm import capture


def methods():
    result={'pytorch':torch.mm,'cublas':gpu_ops.gemm_cublas,'naive':gpu_ops.gemm_naive}
    if hasattr(gpu_ops,'gemm_tiled'):result['tiled']=gpu_ops.gemm_tiled
    if hasattr(gpu_ops,'gemm_register'):
        for i,name in enumerate(['reg64x64','reg32x64','reg16x64','reg16x128','reg32x128','reg64x128']):
            result[name]=functools.partial(gpu_ops.gemm_register,config=i)
    if hasattr(gpu_ops,'gemm_tuned'):result['tuned']=gpu_ops.gemm_tuned
    return result


def run(args):
    path=Path(args.output)
    if path.exists():raise SystemExit('Choose a fresh output path.')
    gpu_ops.initialize();configure_precision()
    torch.manual_seed(20261016);rng=random.Random(20261016)
    selected=methods()
    if args.methods:
        assert not set(args.methods)-selected.keys()
        selected={n:selected[n] for n in args.methods}
    files=['csrc/gemm.cu','csrc/bindings.cpp','gpu_ops/extension.py','gpu_ops/ops.py',
           'reference/gemm.py','scripts/benchmark_gemm.py']
    payload={'environment':collect_environment(),'precision':{'dtype':'float32','allow_tf32':torch.backends.cuda.matmul.allow_tf32,
             'pytorch_matmul_precision':torch.get_float32_matmul_precision(),'direct_cublas':'CUBLAS_COMPUTE_32F_PEDANTIC',
             'cublas_package':importlib.metadata.version('nvidia-cublas-cu12')},
             'iterations':args.iterations,'repeats':args.repeats,'seed':20261016,
             'measurement':'CUDA events around captured repeated calls; compile and Python dispatch excluded',
             'cache_regime':'Fixed input buffers, repeated graph replay; no streaming bandwidth claim',
             'wall_measurement':'Three interleaved synchronized ordinary-call loops including output allocation',
             'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files},'cases':[]}
    for shape in args.shapes:
        m,n,k=map(int,shape.split('x'))
        if min(m,n,k)<1:raise ValueError('Timing dimensions must be positive')
        a=torch.randn(m,k,device='cuda');b=torch.randn(k,n,device='cuda');expected=torch.mm(a,b)
        case={'m':m,'n':n,'k':k,'methods':{},'orders':[]};graphs={}
        for name,fn in selected.items():
            start=time.perf_counter();actual=fn(a,b);torch.cuda.synchronize();first_ms=(time.perf_counter()-start)*1000
            torch.testing.assert_close(actual,expected,**tolerance(k))
            graphs[name]=capture(fn,a,b,args.iterations)
            case['methods'][name]={'first_call_ms':first_ms,'graph_us_samples':[],'wall_us_samples':[],
                                   'max_abs_vs_pytorch':(actual-expected).abs().max().item()}
        for _ in range(args.repeats):
            order=list(selected);rng.shuffle(order);case['orders'].append(order)
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name][0].replay();end.record();end.synchronize()
                case['methods'][name]['graph_us_samples'].append(start.elapsed_time(end)*1000/args.iterations)
        for _ in range(3):
            order=list(selected);rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(args.iterations):actual=selected[name](a,b)
                torch.cuda.synchronize()
                case['methods'][name]['wall_us_samples'].append((time.perf_counter()-start)*1e6/args.iterations)
        for result in case['methods'].values():
            for metric in ['graph','wall']:result[metric+'_us_median']=statistics.median(result[metric+'_us_samples'])
            result['tflops']=2*m*n*k/(result['graph_us_median']*1e6)
        payload['cases'].append(case)
        print(shape,{n:round(r['graph_us_median'],3) for n,r in case['methods'].items()},flush=True)
        del graphs
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--shapes',nargs='+',default=['16x4096x4096','4096x4096x32','512x512x512'])
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--methods',nargs='+');args=p.parse_args()
    if args.iterations<1 or args.repeats<2:p.error('Use positive iterations and at least two repeats')
    with torch.inference_mode():run(args)
