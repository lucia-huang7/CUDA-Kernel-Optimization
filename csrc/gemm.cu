#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <limits>

namespace {
__global__ void gemm_naive_kernel(const float* a,const float* b,float* c,int m,int n,int k) {
  const int row=blockIdx.y*16+threadIdx.y;
  const int col=blockIdx.x*16+threadIdx.x;
  if(row>=m || col>=n)return;
  float sum=0.0f;
  for(int p=0;p<k;++p)sum=fmaf(a[static_cast<int64_t>(row)*k+p],b[static_cast<int64_t>(p)*n+col],sum);
  c[static_cast<int64_t>(row)*n+col]=sum;
}

void check_inputs(const at::Tensor& a,const at::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device()==b.device(),"Inputs must share a CUDA device");
  TORCH_CHECK(a.dim()==2 && b.dim()==2,"Inputs must be matrices");
  TORCH_CHECK(a.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat,"FP32 inputs required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),"Contiguous row-major inputs required");
  TORCH_CHECK(a.size(1)==b.size(0),"Inner dimensions must match");
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad(),"GEMM is forward-only; detach inputs first");
  for(auto value:{a.size(0),a.size(1),b.size(1)})
    TORCH_CHECK(value<=std::numeric_limits<int>::max(),"Dimensions exceed the integer API limit");
  TORCH_CHECK((a.size(0)+15)/16<=65535,"Rows exceed the supported CUDA grid limit");
}

at::Tensor prepare(const at::Tensor& a,const at::Tensor& b) {
  return at::empty({a.size(0),b.size(1)},a.options());
}

bool empty_product(const at::Tensor& a,at::Tensor& c) {
  if(c.numel()==0)return true;
  if(a.size(1)==0) {
    C10_CUDA_CHECK(cudaMemsetAsync(c.data_ptr(),0,c.nbytes(),at::cuda::getCurrentCUDAStream(a.get_device()).stream()));
    return true;
  }
  return false;
}
}  // namespace

at::Tensor gemm_naive_cuda(const at::Tensor& a,const at::Tensor& b) {
  check_inputs(a,b);
  c10::cuda::CUDAGuard guard(a.device());
  auto c=prepare(a,b);
  if(empty_product(a,c))return c;
  const int m=a.size(0),k=a.size(1),n=b.size(1);
  const dim3 grid((static_cast<unsigned int>(n)+15)/16,(static_cast<unsigned int>(m)+15)/16);
  gemm_naive_kernel<<<grid,dim3(16,16),0,at::cuda::getCurrentCUDAStream(a.get_device()).stream()>>>(
      a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),m,n,k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

at::Tensor gemm_cublas_cuda(const at::Tensor& a,const at::Tensor& b) {
  check_inputs(a,b);
  c10::cuda::CUDAGuard guard(a.device());
  auto c=prepare(a,b);
  if(empty_product(a,c))return c;
  const int m=a.size(0),k=a.size(1),n=b.size(1);
  auto handle=at::cuda::getCurrentCUDABlasHandle();
  cublasPointerMode_t previous;
  TORCH_CHECK(cublasGetPointerMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"Cannot query cuBLAS pointer mode");
  TORCH_CHECK(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS pointer mode");
  const float alpha=1.0f,beta=0.0f;
  // Row-major AB is column-major B^T A^T with dimensions N x M x K.
  const auto status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,
      b.data_ptr<float>(),CUDA_R_32F,n,a.data_ptr<float>(),CUDA_R_32F,k,&beta,
      c.data_ptr<float>(),CUDA_R_32F,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
  const auto restore=cublasSetPointerMode(handle,previous);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"cuBLAS GEMM failed with status ",int(status));
  TORCH_CHECK(restore==CUBLAS_STATUS_SUCCESS,"Cannot restore cuBLAS pointer mode");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
