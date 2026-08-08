"""PyTorch-only reproducer for the recorded wide FP64 mean racecheck report."""
import torch

torch.manual_seed(20261016)
x=torch.randn(2,65536,device='cuda',dtype=torch.float64)
y=x.mean(-1)
torch.testing.assert_close(y.cpu(),x.cpu().mean(-1),rtol=1e-12,atol=1e-12)
print('PYTORCH_FP64_MEAN_NUMERICAL_CHECK_PASSED')
