#include <iostream>
#include <cudaviz/cudaviz.hpp>
#include <vector>

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

void matAdd() {
    const int N = 2 << 8;
    float A[N][N];
    float B[N][N];
    float C[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
            C[i][j] = 0.0f;
        }
    }

    std::size_t sz = N * N * sizeof(float);
    float* deviceA;
    float* deviceB;
    float* deviceC;

    cudaMalloc(&deviceA, sz);
    cudaMalloc(&deviceB, sz);
    cudaMalloc(&deviceC, sz);

    cudaMemcpy(deviceA, &A[0][0], sz, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, &B[0][0], sz, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, &C[0][0], sz, cudaMemcpyHostToDevice);

    cudaviz::matAdd(deviceA, deviceB, deviceC, N);

    cudaMemcpy(&A[0][0], deviceA, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(&B[0][0], deviceB, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(&C[0][0], deviceC, sz, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    float error = 0;
    for (int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j) {
            error += 3 - C[i][j];
        }
    }
    std::cout << "Error: " << error << std::endl;
}


void saxpy()
{
    const int N = 2 << 25;
    const std::size_t sz = N * sizeof(float);
    std::vector<float> X(N);
    std::vector<float> Y(N);

    for (int i = 0; i < N; ++i)
    {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    float *deviceX;
    float *deviceY;
    cudaMalloc(&deviceX, sz);
    cudaMalloc(&deviceY, sz);

    cudaMemcpy(deviceX, X.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, Y.data(), sz, cudaMemcpyHostToDevice);

    cudaviz::saxpy(2.0f, deviceX, deviceY, N);

    cudaMemcpy(X.data(), deviceX, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y.data(), deviceY, sz, cudaMemcpyDeviceToHost);

    cudaFree(deviceX);
    cudaFree(deviceY);

    float error = 0;
    for (int i = 0; i < N; ++i)
    {
        error += Y[i] - (2.0f * 1.0f + 2.0f);
    }
    std::cout << "Error: " << error << std::endl;
}

int main()
{
    setDataWithIndex();
    addArrays();
    matAdd();
    saxpy();
}
