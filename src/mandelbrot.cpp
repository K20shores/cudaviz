#include <cudaviz/Mandelbrot>
#include <cudaviz/kernels.hpp>

#include "check_error.hpp"

#include <vector>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

namespace cudaviz
{
    std::vector<std::vector<int>> naive_mandelbrot(int max_iter, int N, float x_center, float y_center, float zoom)
    {
        std::size_t sz = N * N * sizeof(int);
        std::vector<int> grid = std::vector<int>(N * N, 0);

        int *deviceGrid;

        CUDA_CHECK(cudaMalloc(&deviceGrid, sz));
        CUDA_CHECK(cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice));

        naive_mandelbrot(deviceGrid, N, max_iter, x_center, y_center, zoom);

        CUDA_CHECK(cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceGrid));

        std::vector<std::vector<int>> grid2D(N, std::vector<int>(N));
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                grid2D[i][j] = grid[i * N + j];
            }
        }

        return grid2D;
    }

    std::vector<std::vector<int>> julia(int max_iter, int N, float x_center, float y_center, float zoom)
    {
        std::size_t sz = N * N * sizeof(int);
        std::vector<int> grid = std::vector<int>(N * N, 0);

        int *deviceGrid;

        CUDA_CHECK(cudaMalloc(&deviceGrid, sz));
        CUDA_CHECK(cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice));

        cudaviz::julia(deviceGrid, N, max_iter, x_center, y_center, zoom);

        CUDA_CHECK(cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceGrid));

        std::vector<std::vector<int>> grid2D(N, std::vector<int>(N));
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                grid2D[i][j] = grid[i * N + j];
            }
        }

        return grid2D;

    }
}