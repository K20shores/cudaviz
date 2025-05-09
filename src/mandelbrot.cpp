#include <cudaviz/kernels.hpp>

#include <vector>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

namespace cudaviz
{
    std::vector<std::vector<int>> mandelbrot(int max_iter = 1000, int N = 10)
    {
        std::size_t sz = N * N * sizeof(int);
        std::vector<int> grid = std::vector<int>(N * N, 0);

        int *deviceGrid;

        cudaError_t error =  cudaMalloc(&deviceGrid, sz);
        if (error != cudaSuccess)
        {
            throw std::runtime_error("Error allocating device memory: " + std::string(cudaGetErrorString(error)));
        }
        cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice);

        cudaviz::mandelbrotIteration(deviceGrid, N, max_iter);

        cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost);
        cudaFree(deviceGrid);

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