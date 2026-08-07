#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    std::exit(1); \
  } \
} while (0)

__global__ void increment(float* values, int index) {
  if (threadIdx.x == 0) values[index] += 1.0f;
}

int main(int argc, char**) {
  float* values;
  float result = 0;
  CHECK(cudaMalloc(&values, sizeof(float)));
  CHECK(cudaMemset(values, 0, sizeof(float)));
  // A separate negative-control process deliberately accesses an invalid index.
  increment<<<1, 32>>>(values, argc > 1 ? 32 : 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(&result, values, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(values));
  if (argc == 1 && result != 1.0f) return 2;
  std::puts("RUNTIME_PROBE_COMPLETED");
  return 0;
}
