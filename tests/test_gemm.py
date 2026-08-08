"""GEMM precision, tile tails, input contracts, stream order and graph replay."""
import pytest
import torch
import gpu_ops
from reference.gemm import configure_precision,gemm_oracle,tolerance

CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA GPU required')
VARIANTS=['gemm_naive','gemm_cublas']


@pytest.fixture(params=VARIANTS)
def operation(request):
    configure_precision()
    return getattr(gpu_ops,request.param)


def test_cpu_oracle():
    a=torch.tensor([[1.,2.,3.],[-1.,2.,0.]])
    b=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    torch.testing.assert_close(gemm_oracle(a,b),torch.tensor([[4.,5.],[-1.,2.]]))


@CUDA
@pytest.mark.parametrize('m,n,k',[(0,7,5),(3,0,5),(3,7,0),(1,1,1),(3,5,7),(15,17,31),
                                 (16,32,32),(17,65,33),(63,127,15),(65,129,35),(128,128,256),(16,257,1025)])
def test_shapes(operation,m,n,k):
    torch.manual_seed(101)
    a=torch.randn(m,k,device='cuda');b=torch.randn(k,n,device='cuda')
    actual=operation(a,b)
    torch.testing.assert_close(actual,gemm_oracle(a,b),**tolerance(k))
    torch.testing.assert_close(actual,a@b,**tolerance(k))
    assert actual.shape==(m,n) and actual.dtype==torch.float32


@CUDA
@pytest.mark.parametrize('mode',['zero','constant','tiny','large','cancellation'])
def test_magnitudes(operation,mode):
    a=torch.randn(19,65,device='cuda');b=torch.randn(65,37,device='cuda')
    if mode=='zero':a.zero_()
    elif mode=='constant':a.fill_(1);b.fill_(2)
    elif mode=='tiny':a*=1e-4;b*=1e-4
    elif mode=='large':a*=100;b*=100
    elif mode=='cancellation':a.fill_(1);b[::2]=1;b[1::2]=-1
    expected=gemm_oracle(a,b)
    actual=operation(a,b)
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=0.1 if mode=='large' else 5e-5)


@CUDA
@pytest.mark.parametrize('offsets',[(1,0),(0,1),(1,1)])
def test_contiguous_offsets(operation,offsets):
    ao,bo=offsets
    a=torch.randn(17*33+ao,device='cuda')[ao:].reshape(17,33)
    b=torch.randn(33*65+bo,device='cuda')[bo:].reshape(33,65)
    torch.testing.assert_close(operation(a,b),gemm_oracle(a,b),**tolerance(33))


@CUDA
@pytest.mark.parametrize('case',['cpu_a','cpu_b','rank_a','rank_b','stride_a','stride_b','inner',
                                 'half','bfloat16','double','integer','grad_a','grad_b'])
def test_contract(operation,case):
    a=torch.randn(3,5,device='cuda');b=torch.randn(5,7,device='cuda')
    if case=='cpu_a':a=a.cpu()
    elif case=='cpu_b':b=b.cpu()
    elif case=='rank_a':a=a[0]
    elif case=='rank_b':b=b[0]
    elif case=='stride_a':a=a.T.contiguous().T
    elif case=='stride_b':b=b.T.contiguous().T
    elif case=='inner':b=b[:4]
    elif case in ['half','bfloat16','double','integer']:
        dtype={'half':torch.float16,'bfloat16':torch.bfloat16,'double':torch.float64,'integer':torch.int32}[case]
        a=a.to(dtype);b=b.to(dtype)
    elif case=='grad_a':a.requires_grad_()
    elif case=='grad_b':b.requires_grad_()
    with pytest.raises(RuntimeError):operation(a,b)


@CUDA
def test_stream_and_graph(operation):
    a=torch.empty(17,33,device='cuda');b=torch.empty(33,65,device='cuda')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1000000);a.normal_();b.normal_();actual=operation(a,b)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual,gemm_oracle(a,b),**tolerance(33))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=operation(a,b)
    a.normal_();b.normal_();graph.replay()
    torch.testing.assert_close(captured,gemm_oracle(a,b),**tolerance(33))
