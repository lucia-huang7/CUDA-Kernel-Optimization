#include <cuda_runtime.h>
namespace { __global__ void increment(float* x) { if (threadIdx.x == 0) x[0] += 1.0f; }
}
extern "C" int probe() {
    float* x; float result = 0;
    cudaError_t e = cudaMalloc(&x, sizeof(float)); if(e) return e;
    cudaMemset(x, 0, sizeof(float));
    increment<<<1,32>>>(x);
    e=cudaDeviceSynchronize();
    cudaMemcpy(&result,x,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(x);
    return e ? e : (result == 1.0f ? 0 : 999);
}
