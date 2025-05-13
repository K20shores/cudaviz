#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <format>

#include "check_error.hpp"

constexpr int threadsPerBlock = 256;

namespace kernels
{
  __global__ void add(int *a, int *b, int *c, int N)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
      c[tid] = a[tid] + b[tid];
      tid += blockDim.x * gridDim.x;
    }
  }

  __global__ void dot(float *a, float *b, float* c, int N)
  {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
      temp += a[tid] + b[tid];
      cacheIdx += blockIdx.x * gridDim.x;
    }

    cache[cacheIdx] += temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
      if (cacheIdx < i)
      {
        cache[cacheIdx] += cache[cacheIdx + i];
      }
      __syncthreads();
      i /= 2;
    }

    if (cacheIdx == 0) {
      c[blockIdx.x] = cache[0];
    }
  }
}

void device_data()
{
  int device_count;

  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  cudaDeviceProp prop;

  for (int i = 0; i < device_count; ++i)
  {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    std::cout << std::format(
        "Device {}\n"
        "  Name: {}\n"
        "  Total Global Memory: {} bytes\n"
        "  Shared Memory per Block: {} bytes\n"
        "  Registers per Block: {}\n"
        "  Warp Size: {}\n"
        "  Max Threads per Block: {}\n"
        "  Max Threads Dim: ({}, {}, {})\n"
        "  Max Grid Size: ({}, {}, {})\n"
        "  Clock Rate: {} kHz\n"
        "  Compute Capability: {}.{}\n",
        i,
        prop.name,
        prop.totalGlobalMem,
        prop.sharedMemPerBlock,
        prop.regsPerBlock,
        prop.warpSize,
        prop.maxThreadsPerBlock,
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
        prop.clockRate / 1000,
        prop.major, prop.minor);
  }
}

void add()
{
  constexpr int N = 4096;
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  for (int i = 0; i < N; ++i)
  {
    a[i] = -i;
    b[i] = i * i;
  }

  CUDA_CHECK(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&dev_c, N * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));

  kernels::add<<<128, 128>>>(a, b, c, N);

  CUDA_CHECK(cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
  {
    if (a[i] + b[i] != c[i])
    {
      std::cout << std::format("Error: {} != {} + {}\n", c[i], a[i], b[i]);
    }
  }

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
}

#define imin(a, b) a<b?a:b
void dot(){
  constexpr int N = 2 << 20;
  constexpr int blocks = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

  float *a, *b, *partial_c;
  a = new float[N];
  b = new float[N];
  partial_c = new float[blocks];

  for(int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 1.0f / i;
  }

  float *dev_a, *dev_b, *dev_c;

  CUDA_CHECK(cudaMalloc((void**)&dev_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c, blocks * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dev_a, a, N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, b, N, cudaMemcpyHostToDevice));

  kernels::dot<<<blocks, threadsPerBlock>>>(dev_a, dev_b, partial_c, N);

  CUDA_CHECK(cudaMemcpy(partial_c, dev_c, blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float result = 0;
  for(int i = 0; i < blocks; ++i) {
    result += partial_c[i];
  }

  std::cout << std::format("Dot: {}\n", result);
}

int main()
{
  device_data();
  add();
  dot();
}