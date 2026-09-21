#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

constexpr int kUnroll = 8;

__global__ void copy_kernel(const float4 *__restrict__ src, float4 *__restrict__ dst, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    // grid-size guaranteed to divide n; keep kUnroll independent accesses in flight per thread
    size_t iters = n / (stride * kUnroll);
    const float4 *p = src + tid;
    float4 *q = dst + tid;
    for (size_t k = 0; k < iters; k++) {
        float4 regs[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; u++) regs[u] = p[(size_t)k * stride * kUnroll + (size_t)u * stride];
#pragma unroll
        for (int u = 0; u < kUnroll; u++) q[(size_t)k * stride * kUnroll + (size_t)u * stride] = regs[u];
    }
}

float bench_sm(const float4 *src, float4 *dst, size_t n, int iters, int grid) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int i = 0; i < 3; i++) copy_kernel<<<grid, 256>>>(src, dst, n);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) copy_kernel<<<grid, 256>>>(src, dst, n);
    cudaEventRecord(e);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return (float)((double)n * 16 * iters / (ms / 1000.0) / 1e9);
}

int main() {
    size_t bytes = (size_t)1 << 30;
    int iters = 20;

    unsigned char *d, *h_map, *h_map_dev;
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaHostAlloc(&h_map, bytes, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer(&h_map_dev, h_map, 0));

    size_t n = bytes / 16; // float4 elements
    for (int grid : {1024, 4096, 16384}) {
        printf("grid=%d:\n", grid);
        printf("  D2H SM (SM writes to host mem): %.1f GB/s\n",
               bench_sm((const float4 *)d, (float4 *)h_map_dev, n, iters, grid));
        printf("  H2D SM (SM reads host mem):     %.1f GB/s\n",
               bench_sm((const float4 *)h_map_dev, (float4 *)d, n, iters, grid));
    }
    return 0;
}
