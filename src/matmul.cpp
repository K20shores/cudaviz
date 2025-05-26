#include <cudaviz/MatMul>
#include <cudaviz/kernels.hpp>

#include "check_error.hpp"
#include "cuda_buffer.hpp"

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

        size_t sz = N * N * sizeof(float);
        CudaBuffer dev_a(sz);
        CudaBuffer dev_b(sz);
        CudaBuffer dev_c(sz);

        std::vector<float> host_A(N * N, 2.0f);
        std::vector<float> host_B(N * N, 3.0f);
        CUDA_CHECK(cudaMemcpy(dev_a.data(), host_A.data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b.data(), host_B.data(), sz, cudaMemcpyHostToDevice));

        kernels::matmul(dev_a.data_float(), dev_b.data_float(), dev_c.data_float(), N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return elapsed_time;
    }

    float tiled_matmul(int N)
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));

        size_t sz = N * N * sizeof(float);
        CudaBuffer dev_a(sz);
        CudaBuffer dev_b(sz);
        CudaBuffer dev_c(sz);

        std::vector<float> host_A(N * N, 2.0f);
        std::vector<float> host_B(N * N, 3.0f);
        CUDA_CHECK(cudaMemcpy(dev_a.data(), host_A.data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b.data(), host_B.data(), sz, cudaMemcpyHostToDevice));

        kernels::tiled_matmul(dev_a.data_float(), dev_b.data_float(), dev_c.data_float(), N);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return elapsed_time;
    }
}