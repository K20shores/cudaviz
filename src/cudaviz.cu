#include <cudaviz/Mandelbrot>

namespace cudaviz
{
    namespace device
    {

        __device__ float scale(int k, int N, float min, float max) {
            return min + k * ((max - min) / N);
        }

        __global__ void mandelbrotIteration(int* grid, int N, int max_iter) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            double x = 0;
            double y = 0;
            double xnew = 0;
            double ynew = 0;

            double x0 = scale(i, N, -2.0f, 0.47f);
            double y0 = scale(j, N, -1.12f, 1.12f);

            if (i < N && j < N) {
                int index = j*N + i;
                grid[index] = max_iter;
                for(int iter = 0; iter < max_iter; ++iter) {
                    xnew = x * x - y * y + x0;
                    ynew = 2 * x * y + y0;
                    if (xnew * xnew + ynew * ynew > 4) {
                        grid[index] = iter;
                    }
                    x = xnew;
                    y = ynew;
                }
            }
        }
    }

    void mandelbrotIteration(int* grid, int N, int max_iter) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::mandelbrotIteration<<<numBlocks, threadsPerBlock>>>(grid, N, max_iter);
        cudaDeviceSynchronize();
    }
}