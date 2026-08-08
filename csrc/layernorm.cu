#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace {
constexpr int kThreads = 256;

__device__ float shared_sum(float value, float* partial) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = partial[0];
  // All readers must finish before another reduction reuses the same array.
  __syncthreads();
  return result;
}

template <typename T>
__global__ void layernorm_shared_kernel(const T* x, const T* weight, const T* bias,
                                        T* output, int width, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  __shared__ float partial[kThreads];
  // Shift the statistics by a row anchor to preserve small centered values.
  const float anchor = float(x[base]);
  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += kThreads) sum += float(x[base + col]) - anchor;
  const float mean = shared_sum(sum, partial) / width;
  float squares = 0.0f;
  for (int col = threadIdx.x; col < width; col += kThreads) {
    const float centered = (float(x[base + col]) - anchor) - mean;
    squares += centered * centered;
  }
  const float inv = rsqrtf(shared_sum(squares, partial) / width + eps);
  for (int col = threadIdx.x; col < width; col += kThreads) {
    output[base + col] = T(((float(x[base + col]) - anchor) - mean) * inv * float(weight[col]) + float(bias[col]));
  }
}

__device__ __forceinline__ float warp_reduce(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

template<int Threads>
__device__ __forceinline__ float block_reduce(float value, float* partial) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = warp_reduce(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float total = lane < Threads / 32 ? partial[lane] : 0.0f;
    total = warp_reduce(total);
    if (lane == 0) partial[0] = total;
  }
  __syncthreads();
  return partial[0];
}

template<typename T>
__global__ void layernorm_warp_kernel(const T* x, const T* weight, const T* bias,
                                      T* output, int width, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  const float anchor = float(x[base]);
  // Separate arrays avoid overwriting mean statistics before all warps read them.
  __shared__ float partial[2][4];
  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += 128) sum += float(x[base + col]) - anchor;
  const float mean = block_reduce<128>(sum, partial[0]) / width;
  float squares = 0.0f;
  for (int col = threadIdx.x; col < width; col += 128) {
    const float value = (float(x[base + col]) - anchor) - mean;
    squares += value * value;
  }
  const float inv = rsqrtf(block_reduce<128>(squares, partial[1]) / width + eps);
  for (int col = threadIdx.x; col < width; col += 128)
    output[base + col] = T(((float(x[base + col]) - anchor) - mean) * inv * float(weight[col]) + float(bias[col]));
}

template<typename T>
__global__ void layernorm_cached_kernel(const T* x, const T* weight, const T* bias,
                                        T* output, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * 1536;
  const float anchor = float(x[base]);
  __shared__ float partial[2][4];
  float values[12];
  float sum = 0.0f;
  #pragma unroll
  for (int tile = 0; tile < 12; ++tile) {
    values[tile] = float(x[base + threadIdx.x + tile * 128]) - anchor;
    sum += values[tile];
  }
  const float mean = block_reduce<128>(sum, partial[0]) / 1536.0f;
  float squares = 0.0f;
  #pragma unroll
  for (int tile = 0; tile < 12; ++tile) {
    values[tile] -= mean;
    squares += values[tile] * values[tile];
  }
  const float inv = rsqrtf(block_reduce<128>(squares, partial[1]) / 1536.0f + eps);
  #pragma unroll
  for (int tile = 0; tile < 12; ++tile) {
    const int col = threadIdx.x + tile * 128;
    output[base + col] = T(values[tile] * inv * float(weight[col]) + float(bias[col]));
  }
}

template<typename T>
struct alignas(sizeof(T) * 4) LayerPack { T values[4]; };

template<typename T>
__global__ void layernorm_vector_kernel(const T* x, const T* weight, const T* bias,
                                        T* output, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * 1536;
  const float anchor = float(x[base]);
  __shared__ float partial[2][4];
  float values[3][4];
  float sum = 0.0f;
  #pragma unroll
  for (int tile = 0; tile < 3; ++tile) {
    const int col = threadIdx.x * 4 + tile * 512;
    const auto pack = *reinterpret_cast<const LayerPack<T>*>(x + base + col);
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      values[tile][lane] = float(pack.values[lane]) - anchor;
      sum += values[tile][lane];
    }
  }
  const float mean = block_reduce<128>(sum, partial[0]) / 1536.0f;
  float squares = 0.0f;
  #pragma unroll
  for (int tile = 0; tile < 3; ++tile) {
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      values[tile][lane] -= mean;
      squares += values[tile][lane] * values[tile][lane];
    }
  }
  const float inv = rsqrtf(block_reduce<128>(squares, partial[1]) / 1536.0f + eps);
  #pragma unroll
  for (int tile = 0; tile < 3; ++tile) {
    const int col = threadIdx.x * 4 + tile * 512;
    const auto w = *reinterpret_cast<const LayerPack<T>*>(weight + col);
    const auto b = *reinterpret_cast<const LayerPack<T>*>(bias + col);
    LayerPack<T> result;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane)
      result.values[lane] = T(values[tile][lane] * inv * float(w.values[lane]) + float(b.values[lane]));
    *reinterpret_cast<LayerPack<T>*>(output + base + col) = result;
  }
}

void check_layernorm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be contiguous CUDA storage");
  TORCH_CHECK(x.dim() == 2, "Input must have shape [rows, width]");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "Supported dtypes: float32, float16, bfloat16");
  TORCH_CHECK(!x.requires_grad(), "Operator is forward-only; detach input first");
  const int64_t width = x.size(1);
  TORCH_CHECK(width > 0 && width <= 65536, "Width must be in [1, 65536]");
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(), "Rows exceed launch grid limit");
  for (const auto& p : {w, b}) {
    TORCH_CHECK(p.is_cuda() && p.device() == x.device(), "Affine tensors must share the input CUDA device");
    TORCH_CHECK(p.dim() == 1 && p.numel() == width && p.is_contiguous(), "Affine tensors must have contiguous shape [width]");
    TORCH_CHECK(p.scalar_type() == x.scalar_type(), "Affine and input dtypes must match");
    TORCH_CHECK(!p.requires_grad(), "Operator is forward-only; detach affine tensors first");
  }
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && static_cast<float>(eps) > 0 &&
              eps <= std::numeric_limits<float>::max(), "Epsilon must be finite and positive in float32");
}
}  // namespace

namespace {
enum class Variant { Shared, Warp, Cached, Vector };
at::Tensor run_layernorm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
                         double eps, Variant variant) {
  check_layernorm(x, w, b, eps);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x);
  if (x.size(0) == 0) return output;
  const int width = x.size(1);
  const unsigned int rows = x.size(0);
  const int alignment = x.element_size() * 4;
  const bool aligned = reinterpret_cast<uintptr_t>(x.data_ptr()) % alignment == 0 &&
                       reinterpret_cast<uintptr_t>(w.data_ptr()) % alignment == 0 &&
                       reinterpret_cast<uintptr_t>(b.data_ptr()) % alignment == 0;
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "layernorm_cuda", [&] {
    auto xp = x.data_ptr<scalar_t>(); auto wp = w.data_ptr<scalar_t>(); auto bp = b.data_ptr<scalar_t>();
    auto op = output.data_ptr<scalar_t>();
    if (variant == Variant::Shared) {
      layernorm_shared_kernel<scalar_t><<<rows, 256, 0, stream.stream()>>>(xp,wp,bp,op,width,float(eps));
    } else if (width == 1536 && variant == Variant::Vector && aligned) {
      layernorm_vector_kernel<scalar_t><<<rows, 128, 0, stream.stream()>>>(xp,wp,bp,op,float(eps));
    } else if (width == 1536 && (variant == Variant::Cached || variant == Variant::Vector)) {
      layernorm_cached_kernel<scalar_t><<<rows, 128, 0, stream.stream()>>>(xp,wp,bp,op,float(eps));
    } else {
      layernorm_warp_kernel<scalar_t><<<rows, 128, 0, stream.stream()>>>(xp,wp,bp,op,width,float(eps));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

at::Tensor layernorm_cuda(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,double eps) {
  return run_layernorm(x,w,b,eps,Variant::Shared);
}
at::Tensor layernorm_warp_cuda(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,double eps) {
  return run_layernorm(x,w,b,eps,Variant::Warp);
}
at::Tensor layernorm_cached_cuda(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,double eps) {
  return run_layernorm(x,w,b,eps,Variant::Cached);
}
at::Tensor layernorm_vector_cuda(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,double eps) {
  return run_layernorm(x,w,b,eps,Variant::Vector);
}
