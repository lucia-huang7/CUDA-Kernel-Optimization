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

__global__ void gemm_tiled_kernel(const float* a,const float* b,float* c,int m,int n,int k) {
  __shared__ float as[16][16];
  __shared__ float bs[16][16];
  const int row=blockIdx.y*16+threadIdx.y;
  const int col=blockIdx.x*16+threadIdx.x;
  float sum=0.0f;
  for(int start=0;start<k;start+=16) {
    as[threadIdx.y][threadIdx.x]=(row<m && start+threadIdx.x<k)
      ?a[static_cast<int64_t>(row)*k+start+threadIdx.x]:0.0f;
    bs[threadIdx.y][threadIdx.x]=(start+threadIdx.y<k && col<n)
      ?b[static_cast<int64_t>(start+threadIdx.y)*n+col]:0.0f;
    __syncthreads();
    #pragma unroll
    for(int p=0;p<16;++p)sum=fmaf(as[threadIdx.y][p],bs[p][threadIdx.x],sum);
    __syncthreads();
  }
  if(row<m && col<n)c[static_cast<int64_t>(row)*n+col]=sum;
}

template<int BM,int BN,int BK,int TM,int TN>
__global__ void gemm_register_kernel(const float* a,const float* b,float* c,int m,int n,int k) {
  constexpr int Threads=(BM/TM)*(BN/TN);
  constexpr int RowThreads=BM/TM;
  constexpr int ColThreads=BN/TN;
  __shared__ float as[BM][BK+1];
  __shared__ float bs[BK][BN];
  const int tr=threadIdx.x/ColThreads;
  const int tc=threadIdx.x%ColThreads;
  const int block_row=blockIdx.y*BM;
  const int block_col=blockIdx.x*BN;
  float accum[TM][TN]={};
  for(int start=0;start<k;start+=BK) {
    for(int index=threadIdx.x;index<BM*BK;index+=Threads) {
      const int r=index/BK,p=index%BK;
      as[r][p]=(block_row+r<m && start+p<k)
        ?a[static_cast<int64_t>(block_row+r)*k+start+p]:0.0f;
    }
    for(int index=threadIdx.x;index<BK*BN;index+=Threads) {
      const int p=index/BN,col=index%BN;
      bs[p][col]=(start+p<k && block_col+col<n)
        ?b[static_cast<int64_t>(start+p)*n+block_col+col]:0.0f;
    }
    __syncthreads();
    #pragma unroll
    for(int p=0;p<BK;++p) {
      float ar[TM],br[TN];
      #pragma unroll
      for(int i=0;i<TM;++i)ar[i]=as[tr+i*RowThreads][p];
      #pragma unroll
      for(int j=0;j<TN;++j)br[j]=bs[p][tc+j*ColThreads];
      #pragma unroll
      for(int i=0;i<TM;++i) {
        #pragma unroll
        for(int j=0;j<TN;++j)accum[i][j]=fmaf(ar[i],br[j],accum[i][j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<TM;++i) {
    const int row=block_row+tr+i*RowThreads;
    #pragma unroll
    for(int j=0;j<TN;++j) {
      const int col=block_col+tc+j*ColThreads;
      if(row<m && col<n)c[static_cast<int64_t>(row)*n+col]=accum[i][j];
    }
  }
}

template<int BM,int BN,int BK,int TM,int TN>
void launch_register(const at::Tensor& a,const at::Tensor& b,at::Tensor& c,cudaStream_t stream) {
  constexpr int Threads=(BM/TM)*(BN/TN);
  dim3 grid((static_cast<unsigned int>(b.size(1))+BN-1)/BN,
            (static_cast<unsigned int>(a.size(0))+BM-1)/BM);
  gemm_register_kernel<BM,BN,BK,TM,TN><<<grid,Threads,0,stream>>>(
      a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),a.size(0),b.size(1),a.size(1));
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

at::Tensor gemm_tiled_cuda(const at::Tensor& a,const at::Tensor& b) {
  check_inputs(a,b);
  c10::cuda::CUDAGuard guard(a.device());
  auto c=prepare(a,b);
  if(empty_product(a,c))return c;
  const int m=a.size(0),n=b.size(1),k=a.size(1);
  dim3 grid((static_cast<unsigned int>(n)+15)/16,(static_cast<unsigned int>(m)+15)/16);
  gemm_tiled_kernel<<<grid,dim3(16,16),0,at::cuda::getCurrentCUDAStream(a.get_device()).stream()>>>(
      a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),m,n,k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

at::Tensor gemm_register_cuda(const at::Tensor& a,const at::Tensor& b,int64_t config) {
  check_inputs(a,b);
  TORCH_CHECK(config>=0 && config<6,"Register config must be in [0, 5]");
  c10::cuda::CUDAGuard guard(a.device());
  auto c=prepare(a,b);
  if(empty_product(a,c))return c;
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  switch(config) {
    case 0:launch_register<64,64,16,4,4>(a,b,c,stream);break;
    case 1:launch_register<32,64,16,2,4>(a,b,c,stream);break;
    case 2:launch_register<16,64,32,2,4>(a,b,c,stream);break;
    case 3:launch_register<16,128,32,2,8>(a,b,c,stream);break;
    case 4:launch_register<32,128,32,4,4>(a,b,c,stream);break;
    case 5:launch_register<64,128,32,4,4>(a,b,c,stream);break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
