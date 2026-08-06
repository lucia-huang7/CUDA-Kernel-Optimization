"""Small vector-alignment and row-tail workload for Compute Sanitizer."""
import torch
from gpu_ops import rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128
from reference.rmsnorm import rmsnorm_reference

count = 0
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for x_offset, w_offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        x = torch.randn(5 * 1536 + x_offset, device='cuda', dtype=dtype)[x_offset:].reshape(5, 1536)
        w = torch.randn(1536 + w_offset, device='cuda', dtype=dtype)[w_offset:]
        expected = rmsnorm_reference(x, w)
        tol = {'rtol': 1e-5, 'atol': 2e-6} if dtype == torch.float32 else ({'rtol': 2e-3, 'atol': 2e-3} if dtype == torch.float16 else {'rtol': 2e-2, 'atol': 2e-2})
        for operation in [rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128]:
            torch.testing.assert_close(operation(x, w), expected, **tol)
            count += 1
print('PASS:', count, 'specialized alignment/tail cases')
