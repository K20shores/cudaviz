#include <cudaviz/RayTrace>
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
    std::vector<std::vector<RGB>> ray_trace(int N, int n_spheres)
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));

        std::vector<unsigned char> host_data(N * N * 3);
        unsigned char *device_data;

        CUDA_CHECK(cudaMalloc((void **)&device_data, N * N * 3));

        kernels::ray_trace(device_data, N, n_spheres);

        CUDA_CHECK(cudaMemcpy(host_data.data(), device_data, N * N * 3, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

        std::cout << std::format("Generation time: {}\n", elapsed_time);

        CUDA_CHECK(cudaFree(device_data));

        std::vector<std::vector<RGB>> pixels(N, std::vector<RGB>(N));
        for (int y = 0; y < N; ++y)
        {
            for (int x = 0; x < N; ++x)
            {
                pixels[y][x].r = host_data[3 * (N * y + x) + 0];
                pixels[y][x].g = host_data[3 * (N * y + x) + 1];
                pixels[y][x].b = host_data[3 * (N * y + x) + 2];
            }
        }

        return pixels;
    }
}