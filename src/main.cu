#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

// Kernel function to add the elements of two arrays
__global__
void testKernel(float *x)
{
  x[0] = 123.456f;
}

int main(void)
{
  // Query and print GPU properties
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Running on GPU: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

  float *x;
  cudaError_t err;

  // Allocate unified memory and check for errors
  err = cudaMallocManaged(&x, sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  x[0] = 0.0f;

  // Launch kernel and check for errors
  testKernel<<<1, 1>>>(x);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(x);
    return -1;
  }

  // Synchronize and check for errors
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(x);
    return -1;
  }

  std::cout << "x[0] = " << x[0] << std::endl;

  // Free memory
  cudaFree(x);
  return 0;
}