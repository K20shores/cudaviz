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
    std::cout << std::format("Device {}\n", i);
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