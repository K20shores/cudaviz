#pragma once

#include "check_error.hpp"

#include <cuda_runtime.h>

class CudaBuffer {
  public:
    CudaBuffer() = delete;
    CudaBuffer(size_t sz) {
      CUDA_CHECK(cudaMalloc((void**)&mem, sz));
    }

    void* data() { return mem; }
    float* data_float() { return static_cast<float*>(mem); }

    ~CudaBuffer() {
      if (mem) {
        CUDA_CHECK(cudaFree(mem));
        mem = nullptr;
      }
    }
  private:
    void* mem = nullptr;
};