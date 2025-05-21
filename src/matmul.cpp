#include <cudaviz/MatMul>
#include <cudaviz/kernels.hpp>

#include "check_error.hpp"

#include <vector>
#include <string>
#include <stdexcept>
#include <format>
#include <iostream>

#include <cuda_runtime.h>

namespace cudaviz
{
    void matmul(int N)
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));

        float *device_A;
        float *device_B;
        float *device_C;

        CUDA_CHECK(cudaMalloc((void **)&device_A, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&device_B, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&device_C, N * N * sizeof(float)));

        CUDA_CHECK(cudaMemset(device_A, 2, N * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(device_B, 3, N * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(device_C, 0, N * N * sizeof(float)));

        kernels::matmul(device_A, device_B, device_C, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        std::cout << std::format("Matrix multiplication time: {}\n", elapsed_time);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(device_A));
        CUDA_CHECK(cudaFree(device_B));
        CUDA_CHECK(cudaFree(device_C));

        return;
    }
}