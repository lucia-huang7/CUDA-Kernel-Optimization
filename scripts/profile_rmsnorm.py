"""Run one selected warmed RMSNorm kernel for Nsight Compute."""
import argparse
import torch
from gpu_ops import rmsnorm

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=2048)
parser.add_argument('--width', type=int, default=1536)
parser.add_argument('--variant', choices=['shared', 'warp'], default='shared')
args = parser.parse_args()
if args.variant == 'warp':
    from gpu_ops import rmsnorm_warp as operation
else:
    operation = rmsnorm
x = torch.randn(args.rows, args.width, device='cuda', dtype=torch.bfloat16)
w = torch.randn(args.width, device='cuda', dtype=x.dtype)
operation(x, w)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
output = operation(x, w)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print('Profiled', args.variant, args.rows, args.width)
