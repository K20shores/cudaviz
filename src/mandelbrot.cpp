#include <cudaviz/kernels.hpp>

#include <vector>

#include <cuda_runtime.h>

std::vector<int> mandelbrotIteration(int max_iter = 1000, int N = 10) {
    std::size_t sz = N * N * sizeof(int);
    std::vector<int> grid = std::vector<int>(N * N, 0.0f);

    int* deviceGrid;

    cudaMalloc(&deviceGrid, sz);
    cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice);

    cudaviz::mandelbrotIteration(deviceGrid, N, max_iter);

    cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost);
    cudaFree(deviceGrid);

    return grid;
}