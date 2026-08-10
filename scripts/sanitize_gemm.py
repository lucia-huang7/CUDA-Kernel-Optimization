"""Compact GEMM tile-tail/alignment workload with an independent CPU oracle."""
import hashlib
import json
from pathlib import Path
import torch
import gpu_ops
from reference.gemm import configure_precision,gemm_oracle,tolerance
from scripts.benchmark_gemm import methods


def main():
    gpu_ops.initialize();configure_precision();torch.manual_seed(20261016);torch.set_num_threads(4)
    count=0
    for m,n,k in [(17,65,33),(19,132,36),(19,132,260)]:
        for ao,bo in [(0,0),(1,0),(0,1)]:
            a=torch.randn(m*k+ao,device='cuda')[ao:].reshape(m,k)
            b=torch.randn(k*n+bo,device='cuda')[bo:].reshape(k,n)
            expected=gemm_oracle(a,b)
            for name,fn in methods().items():
                torch.testing.assert_close(fn(a,b),expected,**tolerance(k));count+=1
    for m,n,k in [(0,7,5),(3,0,5),(3,7,0)]:
        a=torch.empty(m,k,device='cuda');b=torch.empty(k,n,device='cuda')
        expected=gemm_oracle(a,b)
        for name,fn in methods().items():
            torch.testing.assert_close(fn(a,b),expected,**tolerance(k));count+=1
    torch.cuda.synchronize()
    print('PASS:',count,'GEMM sanitizer cases; all candidates, tile tails, alignment and empty dimensions.')


if __name__=='__main__':main()
