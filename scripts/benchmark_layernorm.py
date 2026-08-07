"""Interleaved LayerNorm replay and ordinary-call measurements on fixed buffers."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
import gpu_ops
from reference.layernorm import layernorm_reference, layernorm_native
from scripts.check_env import collect_environment
from scripts.benchmark_rmsnorm import capture


def run(args):
    path=Path(args.output)
    if path.exists(): raise SystemExit("Choose a fresh output path.")
    gpu_ops.initialize()
    torch.manual_seed(20261016); rng=random.Random(20261016)
    methods={"pytorch_eager":layernorm_reference,"pytorch_native":layernorm_native,"cuda_shared":gpu_ops.layernorm}
    for name in ["warp","cached","vector"]:
        if hasattr(gpu_ops,"layernorm_"+name): methods["cuda_"+name]=getattr(gpu_ops,"layernorm_"+name)
    if args.compiled: methods["pytorch_compiled"]=torch.compile(layernorm_reference,fullgraph=True,dynamic=False)
    if args.methods:
        if set(args.methods)-methods.keys(): raise ValueError("Unknown methods")
        methods={n:methods[n] for n in args.methods}
    sources=["csrc/layernorm.cu","csrc/bindings.cpp","gpu_ops/extension.py","gpu_ops/ops.py",
             "reference/layernorm.py","scripts/benchmark_layernorm.py"]
    payload={"environment":collect_environment(),"seed":20261016,"iterations":args.iterations,"repeats":args.repeats,
             "measurement":"CUDA event timing of repeated CUDA Graph calls; compile/dispatch excluded",
             "cache_regime":"Repeated fixed input and affine buffers, warm cache",
             "wall_measurement":"Three synchronized interleaved Python loops, including output allocation",
             "source_hashes":{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},"cases":[]}
    for rows in args.rows:
        x=torch.randn(rows,args.width,device="cuda",dtype=getattr(torch,args.dtype))
        w=torch.randn(args.width,device="cuda",dtype=x.dtype);b=torch.randn_like(w)
        expected=layernorm_native(x,w,b)
        tol=dict(rtol=2e-5,atol=5e-6) if x.dtype==torch.float32 else (dict(rtol=2e-3,atol=2e-3) if x.dtype==torch.float16 else dict(rtol=2e-2,atol=2e-2))
        case={"rows":rows,"width":args.width,"dtype":args.dtype,"methods":{},"orders":[]};graphs={}
        for name,fn in methods.items():
            start=time.perf_counter();actual=fn(x,w,b);torch.cuda.synchronize()
            first_ms=(time.perf_counter()-start)*1000
            torch.testing.assert_close(actual,expected,**tol)
            bound=lambda xx,ww,fn=fn,bb=b:fn(xx,ww,bb)
            graphs[name]=capture(bound,x,w,args.iterations)
            case["methods"][name]={"first_call_ms":first_ms,"graph_us_samples":[],"wall_us_samples":[],
                                   "max_abs_error":(actual.float()-expected.float()).abs().max().item()}
        for _ in range(args.repeats):
            order=list(methods);rng.shuffle(order);case["orders"].append(order)
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name][0].replay();end.record();end.synchronize()
                case["methods"][name]["graph_us_samples"].append(start.elapsed_time(end)*1000/args.iterations)
        for _ in range(3):
            order=list(methods);rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(args.iterations): actual=methods[name](x,w,b)
                torch.cuda.synchronize()
                case["methods"][name]["wall_us_samples"].append((time.perf_counter()-start)*1e6/args.iterations)
        for result in case["methods"].values():
            for metric in ["graph","wall"]: result[metric+"_us_median"]=statistics.median(result[metric+"_us_samples"])
        payload["cases"].append(case)
        print(rows,args.width,{n:round(r["graph_us_median"],3) for n,r in case["methods"].items()},flush=True)
        del graphs
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    p.add_argument("--rows",type=int,nargs="+",default=[1,8,32,128,512,2048]);p.add_argument("--width",type=int,default=1536)
    p.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="bfloat16")
    p.add_argument("--iterations",type=int,default=100);p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--compiled",action="store_true");p.add_argument("--methods",nargs="+")
    args=p.parse_args()
    if args.iterations<1 or args.repeats<2 or args.width<1 or any(r<1 for r in args.rows):p.error("Use positive dimensions/iterations and at least two repeats")
    with torch.inference_mode():run(args)
