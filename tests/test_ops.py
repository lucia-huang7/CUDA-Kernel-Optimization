"""Forward correctness, input contracts and CUDA execution semantics."""
import pytest
import torch
from gpu_ops import add_scalar, rmsnorm, rmsnorm_warp, rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128
from reference.rmsnorm import rmsnorm_reference

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOLERANCES = {
    torch.float32: {'rtol': 1e-5, 'atol': 2e-6},
    torch.float16: {'rtol': 2e-3, 'atol': 2e-3},
    torch.bfloat16: {'rtol': 2e-2, 'atol': 2e-2},
}


@pytest.fixture(params=[rmsnorm, rmsnorm_warp, rmsnorm_cached, rmsnorm_rows, rmsnorm_vector, rmsnorm_reload, rmsnorm_vector128],
                ids=['shared', 'warp', 'cached', 'rows', 'vector', 'reload', 'vector128'])
def operation(request):
    return request.param


@pytest.mark.parametrize('dtype', DTYPES)
def test_reference_against_float64(dtype):
    torch.manual_seed(17)
    x, w = torch.randn(8, 1537).to(dtype), torch.randn(1537).to(dtype)
    values = x.double()
    expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)).to(dtype) * w
    torch.testing.assert_close(rmsnorm_reference(x, w), expected, **TOLERANCES[dtype])


def test_reference_row_independence():
    x, w = torch.randn(8, 33), torch.randn(33)
    expected = torch.cat([rmsnorm_reference(row[None], w) for row in x])
    torch.testing.assert_close(rmsnorm_reference(x, w), expected)


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('rows,width', [(0, 1536), (1, 1), (1, 1536), (8, 1537), (128, 768), (2048, 1536), (2, 65536)])
def test_rmsnorm_shapes(operation, dtype, rows, width):
    torch.manual_seed(29)
    x = torch.randn(rows, width, device='cuda', dtype=dtype)
    w = torch.randn(width, device='cuda', dtype=dtype)
    actual = operation(x, w)
    torch.testing.assert_close(actual, rmsnorm_reference(x, w), **TOLERANCES[dtype])
    assert actual.dtype == dtype and actual.shape == x.shape


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('scale', [0.0, 1e-7, 1e4])
def test_rmsnorm_magnitudes(operation, dtype, scale):
    torch.manual_seed(37)
    x = torch.randn(8, 1537, device='cuda', dtype=dtype) * scale
    w = torch.randn(1537, device='cuda', dtype=dtype)
    torch.testing.assert_close(operation(x, w), rmsnorm_reference(x, w), **TOLERANCES[dtype])


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('case', ['cpu', 'noncontiguous', 'input_rank', 'weight_shape', 'weight_dtype',
                                  'weight_stride', 'zero_width', 'too_wide', 'eps_zero',
                                  'eps_nan', 'eps_inf', 'eps_underflow', 'input_grad', 'weight_grad', 'integer'])
def test_invalid_input_is_rejected(operation, case):
    x = torch.randn(4, 16, device='cuda')
    w, eps = torch.ones(16, device='cuda'), 1e-6
    if case == 'cpu': x = x.cpu()
    elif case == 'noncontiguous': x = torch.randn(16, 4, device='cuda').T
    elif case == 'input_rank': x = x[0]
    elif case == 'weight_shape': w = w[:15]
    elif case == 'weight_dtype': w = w.half()
    elif case == 'weight_stride': w = torch.ones(32, device='cuda')[::2]
    elif case == 'zero_width': x, w = x[:, :0], w[:0]
    elif case == 'too_wide': x, w = torch.ones(1, 65537, device='cuda'), torch.ones(65537, device='cuda')
    elif case == 'eps_zero': eps = 0
    elif case == 'eps_nan': eps = float('nan')
    elif case == 'eps_inf': eps = float('inf')
    elif case == 'eps_underflow': eps = 1e-100
    elif case == 'input_grad': x.requires_grad_(True)
    elif case == 'weight_grad': w.requires_grad_(True)
    elif case == 'integer': x, w = x.long(), w.long()
    with pytest.raises(RuntimeError):
        operation(x, w, eps)


@CUDA
@pytest.mark.cuda
def test_nondefault_stream_ordering(operation):
    x = torch.zeros(8, 1536, device='cuda')
    w = torch.randn(1536, device='cuda')
    add_scalar(x)
    operation(x, w)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        x.fill_(3.0)
        added = add_scalar(x, 2.0)
        normalized = operation(x, w)
    stream.synchronize()
    torch.testing.assert_close(added, torch.full_like(x, 5.0), rtol=0, atol=0)
    torch.testing.assert_close(normalized, rmsnorm_reference(x, w), **TOLERANCES[x.dtype])


@CUDA
@pytest.mark.cuda
def test_graph_capture_and_updated_input(operation):
    x = torch.randn(8, 1536, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(1536, device='cuda', dtype=x.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = operation(x, w)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(actual, rmsnorm_reference(x, w), **TOLERANCES[x.dtype])


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('rows', [1, 3, 4, 5, 7, 9])
def test_specialized_row_tails(operation, dtype, rows):
    x = torch.randn(rows, 1536, device='cuda', dtype=dtype)
    w = torch.randn(1536, device='cuda', dtype=dtype)
    torch.testing.assert_close(operation(x, w), rmsnorm_reference(x, w), **TOLERANCES[dtype])


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize('dtype', DTYPES)
def test_offset_contiguous_storage(operation, dtype):
    x = torch.randn(8 * 1536 + 1, device='cuda', dtype=dtype)[1:].reshape(8, 1536)
    w = torch.randn(1537, device='cuda', dtype=dtype)[1:]
    assert x.is_contiguous() and w.is_contiguous()
    assert x.data_ptr() % 16 != 0 and w.data_ptr() % 16 != 0
    torch.testing.assert_close(operation(x, w), rmsnorm_reference(x, w), **TOLERANCES[dtype])
