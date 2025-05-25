#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <format>

#include "check_error.hpp"
#include <cudaviz/cudaviz>

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
      temp += a[tid] * b[tid];
      tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;

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

  __global__ void histogram(unsigned char* buffer, int N, unsigned int * hist) {
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    __syncthreads();

    while (i < N) {
      atomicAdd(&(temp[buffer[i]]), 1);
      i += stride;
    }
    __syncthreads();

    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
  }

  __global__ void average(int *a, int *b, int *c, int N)
  {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
      int idx1 = (idx + 1) % 256;
      int idx2 = (idx + 2) % 256;
      float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
      float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
      c[idx] = (as + bs) / 2.0f;
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
        "  Compute Capability: {}.{}\n"
        "  Number of streaming multiprocessor: {}\n"
        "  Maximum number of threads: {}\n",
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
        prop.major, prop.minor,
        prop.multiProcessorCount,
        prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount
      );
  }
}

void add()
{
  constexpr int N = 10;
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

  kernels::add<<<128, 128>>>(dev_a, dev_b, dev_c, N);

  CUDA_CHECK(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

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
float dot_malloc(){
  constexpr int N = 100*1024*1024;
  constexpr int blocks = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

  float *a, *b, *partial_c;
  a = new float[N];
  b = new float[N];
  partial_c = new float[blocks];

  for(int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 2*i;
  }

  float *dev_a, *dev_b, *dev_c;

  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  CUDA_CHECK(cudaMalloc((void**)&dev_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c, blocks * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

  kernels::dot<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

  CUDA_CHECK(cudaMemcpy(partial_c, dev_c, blocks * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  float result = 0;
  for(int i = 0; i < blocks; ++i) {
    result += partial_c[i];
  }

  std::cout << std::format("Dot: {}\n", result);
  float _n = N - 1;
  float expected = 2 * (_n * (_n + 1) * (2 * _n + 1) / 6 );

  float diff = std::abs(result - expected);
  float rel_error = diff / std::abs(expected);

  if (rel_error > 1e-5f) {
    std::cout << std::format("Error: {} != {} (rel error = {})\n", result, expected, rel_error);
  }

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));

  delete[] a;
  delete[] b;
  delete[] partial_c;

  return elapsed;
}

float dot_host_malloc(){
  constexpr int N = 100*1024*1024;
  constexpr int blocks = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

  float *a, *b, *partial_c;
  float *dev_a, *dev_b, *dev_c;

  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaHostAlloc((void**)&a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&partial_c, blocks * sizeof(float), cudaHostAllocMapped));

  CUDA_CHECK(cudaHostGetDevicePointer(&dev_a, a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_b, b, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_c, partial_c, 0));

  for(int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 2*i;
  }

  CUDA_CHECK(cudaEventRecord(start, 0));

  kernels::dot<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  float result = 0;
  for(int i = 0; i < blocks; ++i) {
    result += partial_c[i];
  }

  std::cout << std::format("Dot: {}\n", result);
  float _n = N - 1;
  float expected = 2 * (_n * (_n + 1) * (2 * _n + 1) / 6 );

  float diff = std::abs(result - expected);
  float rel_error = diff / std::abs(expected);

  if (rel_error > 1e-5f) {
    std::cout << std::format("Error: {} != {} (rel error = {})\n", result, expected, rel_error);
  }

  CUDA_CHECK(cudaFreeHost(a));
  CUDA_CHECK(cudaFreeHost(b));
  CUDA_CHECK(cudaFreeHost(partial_c));

  return elapsed;
}

void histogram(){
  constexpr int N = 100 * 1024 * 1024;
  std::vector<unsigned char> buffer(N);
  for(int i = 0; i < N; ++i)
  {
    buffer[i] = rand() % 256;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  unsigned char* dev_buffer;
  unsigned int* dev_hist;

  CUDA_CHECK(cudaMalloc((void**)&dev_buffer, N));
  CUDA_CHECK(cudaMemcpy(dev_buffer, buffer.data(), N, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&dev_hist, 256 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(dev_hist, 0, 256 * sizeof(unsigned int)));

  std::vector<unsigned int> hist(256);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int blocks = prop.multiProcessorCount;

  kernels::histogram<<<blocks*2, 256>>>(dev_buffer, N, dev_hist);

  CUDA_CHECK(cudaMemcpy(hist.data(), dev_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_time;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

  std::cout << std::format("Histogram time: {}\n", elapsed_time);

  for(int i = 0; i < N; ++i) {
    hist[buffer[i]]--;
  }
  for(int i = 0; i < 256; ++i) {
    if (hist[i] != 0) {
      std::cout << std::format("Histogram at {} is not zero!\n", i);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dev_buffer));
  CUDA_CHECK(cudaFree(dev_hist));
}

float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *dev_a;
  float elapsed_time;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<int> a(size);

  CUDA_CHECK(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

  CUDA_CHECK(cudaEventRecord(start, 0));

  for(int i = 0; i < 100; ++i) {
    if (up) {
      CUDA_CHECK(cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice));
    }
    else {
      CUDA_CHECK(cudaMemcpy(a.data(), dev_a, size * sizeof(int), cudaMemcpyHostToDevice));
    }
  }

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_time;
} 

float cuda_host_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));

  CUDA_CHECK(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

  CUDA_CHECK(cudaEventRecord(start, 0));

  for(int i = 0; i < 100; ++i) {
    if (up) {
      CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    }
    else {
      CUDA_CHECK(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyHostToDevice));
    }
  }

  CUDA_CHECK(cudaFreeHost(a));
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_time;
}

void memory_time() {
  constexpr int N = 10*1024*1024;
  float elapsed;
  float MB = 100.0f * float(N) * sizeof(int) / 1024.0f / 1024.0f;

  elapsed = cuda_malloc_test(N, true);
  std::cout << std::format("Time using cuda malloc up: {:3.1f} ms\t{:3.1f} MB/s\n", elapsed, MB/(elapsed/1000));
  elapsed = cuda_malloc_test(N, false);
  std::cout << std::format("Time using cuda malloc down: {:3.1f} ms\t{:3.1f} MB/s\n", elapsed, MB/(elapsed/1000));
  elapsed = cuda_host_malloc_test(N, true);
  std::cout << std::format("Time using cuda host malloc up: {:3.1f} ms\t{:3.1f} MB/s\n", elapsed, MB/(elapsed/1000));
  elapsed = cuda_host_malloc_test(N, false);
  std::cout << std::format("Time using cuda host malloc down: {:3.1f} ms\t{:3.1f} MB/s\n", elapsed, MB/(elapsed/1000));
}

void streams() {
  cudaDeviceProp prop;
  int whichDevice;
  CUDA_CHECK(cudaGetDevice(&whichDevice));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, whichDevice));
  if (!prop.deviceOverlap) {
    std::cout << "Device will not handle overlaps, so no speedup from streams.\n";
  }

  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *a, *b, *c;
  int *dev_a, *dev_b, *dev_c;

  int N = 1024*1024;
  int full_data = N*20;

  CUDA_CHECK(cudaMalloc((void**)&dev_a, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c, N * sizeof(int)));

  CUDA_CHECK(cudaHostAlloc((void**)&a, full_data * sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&b, full_data * sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&c, full_data * sizeof(int), cudaHostAllocDefault));

  for(int i = 0; i < full_data; ++i) {
    a[i] = rand();
    b[i] = rand();
  }

  for(int i = 0; i < full_data; i += N) {
    CUDA_CHECK(cudaMemcpyAsync(dev_a, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_b, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

    kernels::average<<<N/256, 256, 0, stream>>>(dev_a, dev_b, dev_c, N);

    CUDA_CHECK(cudaMemcpyAsync(c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  std::cout << std::format("Stream time: {:3.1f} ms\n", elapsed);

  CUDA_CHECK(cudaFreeHost(a));
  CUDA_CHECK(cudaFreeHost(b));
  CUDA_CHECK(cudaFreeHost(c));

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

void streams_overlapped() {
  cudaDeviceProp prop;
  int whichDevice;
  CUDA_CHECK(cudaGetDevice(&whichDevice));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, whichDevice));
  if (!prop.deviceOverlap) {
    std::cout << "Device will not handle overlaps, so no speedup from streams.\n";
  }

  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  cudaStream_t stream1, stream2;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  int *a, *b, *c;
  int *dev_a0, *dev_b0, *dev_c0;
  int *dev_a1, *dev_b1, *dev_c1;

  int N = 1024*1024;
  int full_data = N*20;

  CUDA_CHECK(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c0, N * sizeof(int)));

  CUDA_CHECK(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

  CUDA_CHECK(cudaHostAlloc((void**)&a, full_data * sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&b, full_data * sizeof(int), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&c, full_data * sizeof(int), cudaHostAllocDefault));

  for(int i = 0; i < full_data; ++i) {
    a[i] = rand();
    b[i] = rand();
  }

  for(int i = 0; i < full_data; i += 2*N) {
    CUDA_CHECK(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream2));

    CUDA_CHECK(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream2));

    kernels::average<<<N/256, 256, 0, stream1>>>(dev_a0, dev_b0, dev_c0, N);
    kernels::average<<<N/256, 256, 0, stream2>>>(dev_a1, dev_b1, dev_c1, N);

    CUDA_CHECK(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream2));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  CUDA_CHECK(cudaEventRecord(stop, 0));

  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

  std::cout << std::format("Stream overlapped time: {:3.1f} ms\n", elapsed);

  CUDA_CHECK(cudaFreeHost(a));
  CUDA_CHECK(cudaFreeHost(b));
  CUDA_CHECK(cudaFreeHost(c));

  CUDA_CHECK(cudaFree(dev_a0));
  CUDA_CHECK(cudaFree(dev_b0));
  CUDA_CHECK(cudaFree(dev_c0));
  CUDA_CHECK(cudaFree(dev_a1));
  CUDA_CHECK(cudaFree(dev_b1));
  CUDA_CHECK(cudaFree(dev_c1));

  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
}

void mapped() {
  cudaDeviceProp prop;
  int whichDevice;
  CUDA_CHECK(cudaGetDevice(&whichDevice));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, whichDevice));
  if (prop.canMapHostMemory != 1) {
    std::cout << "Device cannot map memory\n";
  }

  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  std::cout << std::format("Regular dot: {:3.1f} ms\nPinned dot: {:3.1f} ms\n", dot_malloc(), dot_host_malloc());
}

int main()
{
  // device_data();
  // add();
  // histogram();
  // memory_time();
  // streams();
  // streams_overlapped();
  // mapped();
  std::cout << std::format("Matrix multiplication time: {} ms\n", cudaviz::matmul(4));
}