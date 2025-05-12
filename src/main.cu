#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <format>

#include "check_error.hpp"

__global__ void add(int a, int b, int* c) {
  *c = a + b;
}

void device_data() {
  int device_count;

  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  cudaDeviceProp prop;

  for(int i = 0; i < device_count; ++i) {
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
      prop.major, prop.minor
    );

  }
}

int main () {
  int c;
  int* dev_c;

  CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(int)));

  add<<<1, 1>>>(2, 3, dev_c);

  CUDA_CHECK(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "c: " << c << std::endl;

  CUDA_CHECK(cudaFree(dev_c));
  device_data();
}