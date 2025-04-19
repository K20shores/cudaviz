
#include <iostream>
#include <cudaviz/cudzviz.cuh>

#include <cuda_runtime.h>

int main() {
    const int N = 16;
    int hostData[N];

    // Initialize host data
    for (int i = 0; i < N; ++i)
        hostData[i] = i;

    int *deviceData;
    cudaMalloc(&deviceData, N * sizeof(int));
    cudaMemcpy(deviceData, hostData, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads
    cudaviz::addOneDriver(deviceData);

    // Copy result back to host
    cudaMemcpy(hostData, deviceData, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deviceData);

    // Print results
    std::cout << "Results:\n";
    for (int i = 0; i < N; ++i)
        std::cout << hostData[i] << " ";
    std::cout << std::endl;

    return 0;
}
