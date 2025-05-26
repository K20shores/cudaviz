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
#include <cuda_fp16.h>

using half_t = __half;

namespace cudaviz
{
    float matmul(int N)
    {
        CudaEventBuffer start;
        CudaEventBuffer stop;

        CUDA_CHECK(cudaEventRecord(start.get(), 0));

        size_t sz = N * N * sizeof(float);
        CudaBuffer dev_a(sz);
        CudaBuffer dev_b(sz);
        CudaBuffer dev_c(sz);

        std::vector<float> host_A(N * N, 2.0f);
        std::vector<float> host_B(N * N, 3.0f);
        CUDA_CHECK(cudaMemcpy(dev_a.data(), host_A.data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b.data(), host_B.data(), sz, cudaMemcpyHostToDevice));

        kernels::matmul(dev_a.data_float(), dev_b.data_float(), dev_c.data_float(), N);

        CUDA_CHECK(cudaEventRecord(stop.get(), 0));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start.get(), stop.get()));

        return elapsed_time;
    }

    float tiled_matmul(int N)
    {
        CudaEventBuffer start;
        CudaEventBuffer stop;

        CUDA_CHECK(cudaEventRecord(start.get(), 0));

        size_t sz = N * N * sizeof(float);
        CudaBuffer dev_a(sz);
        CudaBuffer dev_b(sz);
        CudaBuffer dev_c(sz);

        std::vector<float> host_A(N * N, 2.0f);
        std::vector<float> host_B(N * N, 3.0f);
        CUDA_CHECK(cudaMemcpy(dev_a.data(), host_A.data(), sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b.data(), host_B.data(), sz, cudaMemcpyHostToDevice));

        kernels::tiled_matmul(dev_a.data_float(), dev_b.data_float(), dev_c.data_float(), N);

        CUDA_CHECK(cudaEventRecord(stop.get(), 0));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start.get(), stop.get()));

        return elapsed_time;
    }

    float tensor_matmul(int N)
    {
        CudaEventBuffer start;
        CudaEventBuffer stop;

        CUDA_CHECK(cudaEventRecord(start.get(), 0));

        size_t in_sz = N * N * sizeof(half_t);
        size_t out_sz = N * N * sizeof(float);
        CudaBuffer dev_a(in_sz);
        CudaBuffer dev_b(in_sz);
        CudaBuffer dev_c(out_sz);

        std::vector<half_t> host_A(N * N);
        std::vector<half_t> host_B(N * N);
        for (int i = 0; i < N * N; ++i) {
            host_A[i] = half_t(2.0f);
            host_B[i] = half_t(3.0f);
        }
        CUDA_CHECK(cudaMemcpy(dev_a.data(), host_A.data(), in_sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b.data(), host_B.data(), in_sz, cudaMemcpyHostToDevice));

        kernels::tensor_matmul(dev_a.data_half_float(), dev_b.data_half_float(), dev_c.data_float(), N);

        CUDA_CHECK(cudaEventRecord(stop.get(), 0));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start.get(), stop.get()));

        return elapsed_time;
    }
}