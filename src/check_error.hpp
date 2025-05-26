#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

inline cudaError_t cudaCheck(cudaError_t result, const char *file, int line) {
  if (result != cudaSuccess) {
      throw std::runtime_error(
          std::string("CUDA Runtime Error at ") + file + ":" + std::to_string(line) +
          ": " + cudaGetErrorString(result));
  }
  return result;
}