#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

at::Tensor rmsnorm_warp_cuda(const at::Tensor&, const at::Tensor&, double);

namespace {
constexpr int kWidth = 1536;
constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float sum_warp(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void rmsnorm_cached_kernel(const scalar_t* input, const scalar_t* weight,
                                      scalar_t* output, float eps) {
  const int64_t start = static_cast<int64_t>(blockIdx.x) * kWidth;
  constexpr int kValues = kWidth / kBlockThreads;
  float values[kValues];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kValues; ++i) {
    values[i] = static_cast<float>(input[start + threadIdx.x + i * kBlockThreads]);
    sum += values[i] * values[i];
  }
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  __shared__ float partial[kBlockThreads / 32];
  sum = sum_warp(sum);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kBlockThreads / 32 ? partial[lane] : 0.0f;
    total = sum_warp(total);
    if (lane == 0) partial[0] = total;
  }
  __syncthreads();
  const float inverse_rms = rsqrtf(partial[0] / kWidth + eps);
#pragma unroll
  for (int i = 0; i < kValues; ++i) {
    const int col = threadIdx.x + i * kBlockThreads;
    const scalar_t normalized = static_cast<scalar_t>(values[i] * inverse_rms);
    output[start + col] = static_cast<scalar_t>(
        static_cast<float>(normalized) * static_cast<float>(weight[col]));
  }
}

template <typename scalar_t, int N>
struct alignas(16) Pack {
  scalar_t values[N];
};

template <typename scalar_t, bool Vectorized>
__global__ void rmsnorm_rows_kernel(const scalar_t* input, const scalar_t* weight,
                                    scalar_t* output, int64_t rows, float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.y;
  // Each rejected row occupies an entire warp; no block-wide barrier is used.
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const int64_t start = row * kWidth;
  constexpr int kValues = kWidth / 32;
  float values[kValues];
  float sum = 0.0f;
  if constexpr (Vectorized) {
    constexpr int kPack = 16 / sizeof(scalar_t);
    constexpr int kTiles = kWidth / (32 * kPack);
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
      const int col = (tile * 32 + lane) * kPack;
      const auto loaded = *reinterpret_cast<const Pack<scalar_t, kPack>*>(input + start + col);
#pragma unroll
      for (int i = 0; i < kPack; ++i) {
        const float value = static_cast<float>(loaded.values[i]);
        values[tile * kPack + i] = value;
        sum += value * value;
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kValues; ++i) {
      values[i] = static_cast<float>(input[start + lane + i * 32]);
      sum += values[i] * values[i];
    }
  }
  sum = sum_warp(sum);
  // Only lane zero holds the complete reduction result.
  const float inverse_rms = rsqrtf(__shfl_sync(0xffffffffu, sum, 0) / kWidth + eps);
  if constexpr (Vectorized) {
    constexpr int kPack = 16 / sizeof(scalar_t);
    constexpr int kTiles = kWidth / (32 * kPack);
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
      const int col = (tile * 32 + lane) * kPack;
      const auto weights = *reinterpret_cast<const Pack<scalar_t, kPack>*>(weight + col);
      Pack<scalar_t, kPack> result;
#pragma unroll
      for (int i = 0; i < kPack; ++i) {
        const scalar_t normalized = static_cast<scalar_t>(values[tile * kPack + i] * inverse_rms);
        result.values[i] = static_cast<scalar_t>(
            static_cast<float>(normalized) * static_cast<float>(weights.values[i]));
      }
      *reinterpret_cast<Pack<scalar_t, kPack>*>(output + start + col) = result;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kValues; ++i) {
      const int col = lane + i * 32;
      const scalar_t normalized = static_cast<scalar_t>(values[i] * inverse_rms);
      output[start + col] = static_cast<scalar_t>(
          static_cast<float>(normalized) * static_cast<float>(weight[col]));
    }
  }
}

at::Tensor run_specialized(const at::Tensor& input, const at::Tensor& weight,
                           double eps, int variant) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "Input must be contiguous [rows, width]");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16, "Supported dtypes: float32, float16, bfloat16");
  TORCH_CHECK(!input.requires_grad() && !weight.requires_grad(), "Operator is forward-only; detach tensors first");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) <= 65536, "Width must be in [1, 65536]");
  TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(), "Weight must share the input CUDA device");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(1) && weight.is_contiguous(),
              "Weight must be contiguous with shape [width]");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "Input and weight dtypes must match");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && static_cast<float>(eps) > 0 &&
              eps <= std::numeric_limits<float>::max(), "Epsilon must be finite and positive in float32");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max(), "Rows exceed launch grid limit");
  if (input.size(1) != kWidth) return rmsnorm_warp_cuda(input, weight, eps);
  // Contiguous slices can still have a storage offset that breaks 16-byte alignment.
  if (variant == 2 && (reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 != 0 ||
                       reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 != 0)) {
    variant = 1;
  }
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  if (input.size(0) == 0) return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                 input.scalar_type(), "rmsnorm_specialized_cuda", [&] {
    if (variant == 0) {
      rmsnorm_cached_kernel<scalar_t><<<static_cast<unsigned int>(input.size(0)), kBlockThreads, 0, stream.stream()>>>(
          input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), static_cast<float>(eps));
    } else {
      const auto blocks = static_cast<unsigned int>((input.size(0) + kWarpsPerBlock - 1) / kWarpsPerBlock);
      const dim3 threads(32, kWarpsPerBlock);
      if (variant == 2) {
        rmsnorm_rows_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.size(0), static_cast<float>(eps));
      } else {
        rmsnorm_rows_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.size(0), static_cast<float>(eps));
      }
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

at::Tensor rmsnorm_cached_cuda(const at::Tensor& x, const at::Tensor& w, double eps) {
  return run_specialized(x, w, eps, 0);
}
at::Tensor rmsnorm_rows_cuda(const at::Tensor& x, const at::Tensor& w, double eps) {
  return run_specialized(x, w, eps, 1);
}
at::Tensor rmsnorm_vector_cuda(const at::Tensor& x, const at::Tensor& w, double eps) {
  return run_specialized(x, w, eps, 2);
}
