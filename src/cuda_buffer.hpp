#pragma once

#include "check_error.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using half_t = __half;

class CudaBuffer {
  public:
    CudaBuffer() = delete;
    CudaBuffer(size_t sz) {
      CUDA_CHECK(cudaMalloc((void**)&mem, sz));
    }

    void* data() { return mem; }
    float* data_float() { return static_cast<float*>(mem); }
    half_t* data_half_float() { return static_cast<half_t*>(mem); }

    ~CudaBuffer() {
      if (mem) {
        CUDA_CHECK(cudaFree(mem));
        mem = nullptr;
      }
    }
  private:
    void* mem = nullptr;
};

class CudaEventBuffer {
  public:
    CudaEventBuffer() {
      CUDA_CHECK(cudaEventCreate(&event));
    }

    cudaEvent_t& get() { return event; }

    ~CudaEventBuffer() {
      CUDA_CHECK(cudaEventDestroy(event));
    }
  private:
    cudaEvent_t event;
};