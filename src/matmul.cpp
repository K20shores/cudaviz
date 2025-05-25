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
    float matmul(int N)
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

        std::vector<float> host_A(N * N, 2.0f);
        std::vector<float> host_B(N * N, 3.0f);
        CUDA_CHECK(cudaMemcpy(device_A, host_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_B, host_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        kernels::matmul(device_A, device_B, device_C, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(device_A));
        CUDA_CHECK(cudaFree(device_B));
        CUDA_CHECK(cudaFree(device_C));

        return elapsed_time;
    }

    float tiled_matmul(int N)
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

        kernels::tiled_matmul(device_A, device_B, device_C, N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(device_A));
        CUDA_CHECK(cudaFree(device_B));
        CUDA_CHECK(cudaFree(device_C));

        return elapsed_time;
    }
}