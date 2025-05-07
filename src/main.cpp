#include <iostream>
#include <cudaviz/cudaviz.hpp>

#include <cuda_runtime.h>

void setDataWithIndex()
{
    const int N = 16;
    int hostData[N];

    // Initialize host data
    for (int i = 0; i < N; ++i)
        hostData[i] = 0;

    int *deviceData;
    cudaMalloc(&deviceData, N * sizeof(int));
    cudaMemcpy(deviceData, hostData, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads
    cudaviz::setIndex(deviceData);

    // Copy result back to host
    cudaMemcpy(hostData, deviceData, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(deviceData);

    // Print results
    std::cout << "Results:\n";
    for (int i = 0; i < N; ++i)
        std::cout << hostData[i] << " ";
    std::cout << std::endl;
}

void addArrays()
{
    const int N = 2 << 15;
    const std::size_t sz = N * sizeof(float);
    float A[N];
    float B[N];
    float C[N];

    for (int i = 0; i < N; ++i)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 3.0;
    }

    float *deviceA;
    float *deviceB;
    float *deviceC;
    cudaMalloc(&deviceA, sz);
    cudaMalloc(&deviceB, sz);
    cudaMalloc(&deviceC, sz);

    cudaMemcpy(deviceA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, C, sz, cudaMemcpyHostToDevice);

    cudaviz::add(deviceA, deviceB, deviceC, N);

    cudaMemcpy(A, deviceA, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, deviceB, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, deviceC, sz, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    float error = 0;
    for (int i = 0; i < N; ++i)
    {
        error += 3 - C[i];
    }
    std::cout << "Error: " << error << std::endl;
}

int main()
{
    setDataWithIndex();
    addArrays();
}
