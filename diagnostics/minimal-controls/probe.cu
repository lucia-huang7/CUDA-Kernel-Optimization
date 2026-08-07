#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)
__global__ void increment(float* x) { if (threadIdx.x == 0) x[0] += 1.0f; }
int main() {
    float* x; float result = 0;
    CHECK(cudaMalloc(&x, sizeof(float)));
    CHECK(cudaMemset(x, 0, sizeof(float)));
    increment<<<1, 32>>>(x);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&result, x, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(x));
    if (result != 1.0f) return 2;
    std::puts("STANDALONE_CUDA_PASSED");
    return 0;
}
