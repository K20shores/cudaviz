#include <iostream>
#include <cudaviz/cudaviz.hpp>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>

std::vector<int> mandelbrotIteration(int N = 10) {
    std::size_t sz = N * N * sizeof(int);
    std::vector<int> grid = std::vector<int>(N * N, 0.0f);

    int* deviceGrid;

    cudaMalloc(&deviceGrid, sz);
    cudaMemcpy(deviceGrid, grid.data(), sz, cudaMemcpyHostToDevice);

    cudaviz::mandelbrotIteration(deviceGrid, N, 1000);

    cudaMemcpy(grid.data(), deviceGrid, sz, cudaMemcpyDeviceToHost);
    cudaFree(deviceGrid);

    return grid;
}

template<typename T>
void writeGridToFile(const std::vector<T>& grid, int N, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outFile << grid[i * N + j];
            if (j < N - 1) outFile << ",";
        }
        outFile << "\n";
    }

    outFile.close();
}

int main()
{
    int N = 1000;
    auto data = mandelbrotIteration(N);
    writeGridToFile(data, N, "data.csv");
}
