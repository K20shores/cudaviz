#include <cudaviz/cudaviz.hpp>

namespace cudaviz
{
    namespace device
    {
        __global__ void setIndex(int *data)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            data[idx] = idx;
        }

        __global__ void add(float *A, float *B, float *C)
        {
            int i = threadIdx.x;
            C[i] = A[i] + B[i];
            ;
        }

        __global__ void matAdd(float* A, float* B, float* C, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < N && j < N) {
                int index = i*N + j;
                C[index] = A[index] + B[index];
            }
        }

        __global__ void saxpy(float a, float* x, float* y, int N) {
            int thread = blockDim.x * blockIdx.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for(int i = thread; i < N; i += stride)
                y[i] = a * x[i] + y[i];
        }
    }

    void setIndex(int *data)
    {
        device::setIndex<<<4, 4>>>(data);
        cudaDeviceSynchronize();
    }

    void add(float *A, float *B, float *C, int N)
    {
        device::add<<<1, N>>>(A, B, C);
        cudaDeviceSynchronize();
    }

    void matAdd(float* A, float* B, float* C, int N) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::matAdd<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        cudaDeviceSynchronize();
    }


    void saxpy(float a, float* x, float* y, int N) {
        int threadsPerBlock = 256;
        int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        device::saxpy<<<numBlocks, threadsPerBlock>>>(a, x, y, N);
        cudaDeviceSynchronize();
    }
}