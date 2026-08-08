"""LayerNorm semantics, numerical stability, contracts and CUDA ordering."""
import pytest
import torch
from gpu_ops import layernorm, layernorm_warp, layernorm_cached, layernorm_vector
from reference.layernorm import layernorm_reference, layernorm_native

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {torch.float32: dict(rtol=2e-5, atol=5e-6),
       torch.float16: dict(rtol=2e-3, atol=2e-3),
       torch.bfloat16: dict(rtol=2e-2, atol=2e-2)}
VARIANTS = [layernorm, layernorm_warp, layernorm_cached, layernorm_vector]


def oracle(x, w, b, eps=1e-5):
    values = x.cpu().double()
    centered = values - values.mean(-1, keepdim=True)
    return (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
            * w.cpu().double() + b.cpu().double()).to(dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reference(dtype):
    torch.manual_seed(71)
    x, w, b = torch.randn(5, 1537).to(dtype), torch.randn(1537).to(dtype), torch.randn(1537).to(dtype)
    torch.testing.assert_close(layernorm_reference(x, w, b), oracle(x, w, b), **TOL[dtype])


@CUDA
@pytest.mark.parametrize("operation", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,width", [(0, 1536), (1, 1), (1, 31), (3, 33), (8, 256),
                                         (7, 768), (5, 1536), (9, 1537), (2, 4096), (2, 65536)])
def test_shapes(operation, dtype, rows, width):
    torch.manual_seed(73)
    x, w, b = (torch.randn(rows, width, device="cuda", dtype=dtype),
               torch.randn(width, device="cuda", dtype=dtype), torch.randn(width, device="cuda", dtype=dtype))
    actual = operation(x, w, b)
    torch.testing.assert_close(actual, oracle(x, w, b), **TOL[dtype])
    torch.testing.assert_close(actual, layernorm_native(x, w, b), **TOL[dtype])
    assert actual.dtype == dtype and actual.shape == x.shape


@CUDA
@pytest.mark.parametrize("operation", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["zero", "constant", "tiny", "large", "shifted"])
def test_stability(operation, dtype, case):
    torch.manual_seed(79)
    x = torch.randn(5, 1536, device="cuda", dtype=dtype)
    if case == "zero": x.zero_()
    elif case == "constant": x.fill_(10000)
    elif case == "tiny": x *= 1e-7
    elif case == "large": x *= 1e4
    elif case == "shifted": x = x * 0.01 + 10
    w, b = torch.randn(1536, device="cuda", dtype=dtype), torch.randn(1536, device="cuda", dtype=dtype)
    torch.testing.assert_close(operation(x, w, b), oracle(x, w, b), **TOL[dtype])


@CUDA
@pytest.mark.parametrize("operation", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offsets", [(1,0,0), (0,1,0), (0,0,1), (1,1,1)])
def test_storage_offsets(operation, dtype, offsets):
    xo, wo, bo = offsets
    x = torch.randn(5*1536+xo, device="cuda", dtype=dtype)[xo:].reshape(5,1536)
    w = torch.randn(1536+wo, device="cuda", dtype=dtype)[wo:]
    b = torch.randn(1536+bo, device="cuda", dtype=dtype)[bo:]
    torch.testing.assert_close(operation(x,w,b), oracle(x,w,b), **TOL[dtype])


@CUDA
@pytest.mark.parametrize("operation", VARIANTS)
@pytest.mark.parametrize("case", ["cpu", "rank", "stride", "dtype", "empty_width", "too_wide", "weight_shape",
                                  "weight_dtype", "weight_stride", "bias_shape", "bias_dtype", "bias_stride",
                                  "eps_zero", "eps_nan", "eps_inf", "eps_small", "input_grad", "weight_grad", "bias_grad"])
def test_contract(operation, case):
    x = torch.randn(3,16,device="cuda"); w = torch.ones(16,device="cuda"); b = torch.zeros_like(w); eps=1e-5
    if case == "cpu": x=x.cpu()
    elif case == "rank": x=x[0]
    elif case == "stride": x=x.T.contiguous().T
    elif case == "dtype": x=x.int()
    elif case == "empty_width": x=x[:,:0]; w=w[:0]; b=b[:0]
    elif case == "too_wide": x=torch.empty(1,65537,device="cuda")
    elif case == "weight_shape": w=w[:15]
    elif case == "weight_dtype": w=w.half()
    elif case == "weight_stride": w=torch.ones(32,device="cuda")[::2]
    elif case == "bias_shape": b=b[:15]
    elif case == "bias_dtype": b=b.half()
    elif case == "bias_stride": b=torch.zeros(32,device="cuda")[::2]
    elif case == "eps_zero": eps=0
    elif case == "eps_nan": eps=float("nan")
    elif case == "eps_inf": eps=float("inf")
    elif case == "eps_small": eps=1e-100
    elif case == "input_grad": x.requires_grad_()
    elif case == "weight_grad": w.requires_grad_()
    elif case == "bias_grad": b.requires_grad_()
    with pytest.raises(RuntimeError): operation(x,w,b,eps)


@CUDA
@pytest.mark.parametrize("operation", VARIANTS)
def test_stream_and_graph(operation):
    x=torch.empty(5,1536,device="cuda");w=torch.ones(1536,device="cuda");b=torch.zeros_like(w)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1000000)
        x.normal_();actual=operation(x,w,b)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, oracle(x,w,b), **TOL[x.dtype])
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured=operation(x,w,b)
    x.normal_();graph.replay()
    torch.testing.assert_close(captured,oracle(x,w,b),**TOL[x.dtype])
