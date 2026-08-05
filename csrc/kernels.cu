#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace {
constexpr int kThreads = 256;

void check_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16, "Supported dtypes: float32, float16, bfloat16");
  TORCH_CHECK(!input.requires_grad(), "Operators are forward-only; detach input first");
}

template <typename scalar_t>
__global__ void add_scalar_kernel(const scalar_t* input, scalar_t* output,
                                  int64_t count, float value) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<scalar_t>(static_cast<float>(input[index]) + value);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* input, const scalar_t* weight,
                               scalar_t* output, int64_t width, float eps) {
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * width;
  float sum = 0.0f;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = static_cast<float>(input[row_start + col]);
    sum += value * value;
  }
  __shared__ float partial[kThreads];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(width) + eps);
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    // Qwen2 rounds normalized activations to the input dtype before weight scaling.
    const scalar_t normalized = static_cast<scalar_t>(
        static_cast<float>(input[row_start + col]) * inverse_rms);
    output[row_start + col] = static_cast<scalar_t>(
        static_cast<float>(normalized) * static_cast<float>(weight[col]));
  }
}
__device__ __forceinline__ float warp_sum(float value) {
  // All 32 lanes participate, including lanes whose local contribution is zero.
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void rmsnorm_warp_kernel(const scalar_t* input, const scalar_t* weight,
                                    scalar_t* output, int64_t width, float eps) {
  const int64_t row_start = static_cast<int64_t>(blockIdx.x) * width;
  float sum = 0.0f;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = static_cast<float>(input[row_start + col]);
    sum += value * value;
  }
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  __shared__ float partial[kThreads / 32];
  sum = warp_sum(sum);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < kThreads / 32 ? partial[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) partial[0] = block_sum;
  }
  __syncthreads();
  const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(width) + eps);
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    const scalar_t normalized = static_cast<scalar_t>(
        static_cast<float>(input[row_start + col]) * inverse_rms);
    output[row_start + col] = static_cast<scalar_t>(
        static_cast<float>(normalized) * static_cast<float>(weight[col]));
  }
}
}  // namespace

at::Tensor add_scalar_cuda(const at::Tensor& input, double value) {
  check_input(input);
  TORCH_CHECK(std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max(),
              "Scalar must be finite and representable as float32");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  const int64_t count = input.numel();
  if (count == 0) return output;
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "Input exceeds launch grid limit");
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                 input.scalar_type(), "add_scalar_cuda", [&] {
    add_scalar_kernel<scalar_t><<<static_cast<unsigned int>(blocks), kThreads, 0, stream.stream()>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), count, static_cast<float>(value));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor rmsnorm_cuda(const at::Tensor& input, const at::Tensor& weight, double eps) {
  check_input(input);
  TORCH_CHECK(input.dim() == 2, "Input must have shape [rows, width]");
  const int64_t width = input.size(1);
  TORCH_CHECK(width > 0 && width <= 65536, "Width must be in [1, 65536]");
  TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(), "Weight must share the input CUDA device");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == width && weight.is_contiguous(),
              "Weight must be contiguous with shape [width]");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "Input and weight dtypes must match");
  TORCH_CHECK(!weight.requires_grad(), "Operator is forward-only; detach weight first");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && static_cast<float>(eps) > 0 &&
              eps <= std::numeric_limits<float>::max(), "Epsilon must be finite and positive in float32");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max(), "Rows exceed launch grid limit");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  if (input.size(0) == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                 input.scalar_type(), "rmsnorm_cuda", [&] {
    rmsnorm_kernel<scalar_t><<<static_cast<unsigned int>(input.size(0)), kThreads, 0, stream.stream()>>>(
        input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        width, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor rmsnorm_warp_cuda(const at::Tensor& input, const at::Tensor& weight, double eps) {
  check_input(input);
  TORCH_CHECK(input.dim() == 2, "Input must have shape [rows, width]");
  const int64_t width = input.size(1);
  TORCH_CHECK(width > 0 && width <= 65536, "Width must be in [1, 65536]");
  TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(), "Weight must share the input CUDA device");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == width && weight.is_contiguous(),
              "Weight must be contiguous with shape [width]");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "Input and weight dtypes must match");
  TORCH_CHECK(!weight.requires_grad(), "Operator is forward-only; detach weight first");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && static_cast<float>(eps) > 0 &&
              eps <= std::numeric_limits<float>::max(), "Epsilon must be finite and positive in float32");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max(), "Rows exceed launch grid limit");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  if (input.size(0) == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                 input.scalar_type(), "rmsnorm_warp_cuda", [&] {
    rmsnorm_warp_kernel<scalar_t><<<static_cast<unsigned int>(input.size(0)), kThreads, 0, stream.stream()>>>(
        input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        width, static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
