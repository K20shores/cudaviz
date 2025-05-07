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
}