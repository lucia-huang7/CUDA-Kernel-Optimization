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

at::Tensor layernorm_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps) {
  check_layernorm(x, w, b, eps);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x);
  if (x.size(0) == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "layernorm_cuda", [&] {
    layernorm_shared_kernel<scalar_t><<<static_cast<unsigned int>(x.size(0)), kThreads, 0, stream.stream()>>>(
        x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), x.size(1), static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
