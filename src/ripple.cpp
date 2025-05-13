#include <cudaviz/Mandelbrot>
#include <cudaviz/kernels.hpp>

#include "check_error.hpp"

#include <vector>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

namespace cudaviz
{
    std::vector<std::vector<float>> ripple(int N, int tick)
    {
        std::size_t sz = N * N * sizeof(float);
        std::vector<float> grid = std::vector<float>(N * N, 0);

        float *deviceGrid;

        CUDA_CHECK(cudaMalloc(&deviceGrid, sz));
        CUDA_CHECK(cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice));

        _ripple(deviceGrid, N, tick);

        CUDA_CHECK(cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceGrid));

        std::vector<std::vector<float>> grid2D(N, std::vector<float>(N));
        for (int j = 0; j < N; ++j)
        {
            for (int i = 0; i < N; ++i)
            {
                grid2D[j][i] = grid[j * N + i];
            }
        }

        return grid2D;
    }
}