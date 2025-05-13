#include <cudaviz/kernels.hpp>

#include <iostream>

namespace cudaviz
{
    namespace device
    {
        struct cuComplex
        {
            float r;
            float i;
            __device__ cuComplex(float r, float i) : r(r), i(i) {};
            __device__ float magnitude2() { return (r * r + i * i); }
            __device__ cuComplex operator*(const cuComplex &a)
            {
                return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
            }
            __device__ cuComplex operator+(const cuComplex &a)
            {
                return cuComplex(r + a.r, i + a.i);
            }
        };

        __device__ float scale(int k, int N, float min, float max)
        {
            return min + k * ((max - min) / N);
        }

        __global__ void naive_mandelbrot(int *grid, int N, int max_iter, float x_center, float y_center, float zoom)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            float scaled_x_width = 2.47 * zoom;
            float scaled_y_height = 2.24 * zoom;
            double x = 0;
            double y = 0;
            double xnew = 0;
            double ynew = 0;

            double x0 = scale(i, N, x_center - scaled_x_width / 2, x_center + scaled_x_width / 2);
            double y0 = scale(j, N, y_center - scaled_y_height / 2, y_center + scaled_y_height / 2);

            if (i < N && j < N)
            {
                int index = j * N + i;
                grid[index] = max_iter;
                for (int iter = 0; iter < max_iter; ++iter)
                {
                    xnew = x * x - y * y + x0;
                    ynew = 2 * x * y + y0;
                    if (xnew * xnew + ynew * ynew > 4)
                    {
                        grid[index] = iter;
                    }
                    x = xnew;
                    y = ynew;
                }
            }
        }

        __global__ void julia(int *grid, int N, int max_iter, float zoom, float x_center, float y_center)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            float scaled_x_width = 2.47 * zoom;
            float scaled_y_height = 2.24 * zoom;
            double x0 = scale(i, N, x_center - scaled_x_width / 2, x_center + scaled_x_width / 2);
            double y0 = scale(j, N, y_center - scaled_y_height / 2, y_center + scaled_y_height / 2);

            cuComplex c(-0.8, 0.156);
            cuComplex z(x0, y0);

            if (i < N && j < N)
            {
                int index = j * N + i;
                grid[index] = max_iter;
                for (int iter = 0; iter < max_iter; ++iter)
                {
                    z = z * z + c;
                    if (z.magnitude2() > 1000)
                    {
                        grid[index] = iter;
                        break;
                    }
                }
            }
        }

        __global__ void naive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float diffusion_number)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int offset = y * nx + x;

            if (x < nx && y < ny)
            {
                int idx_left = offset - 1;
                int idx_right = offset + 1;

                int idx_top = offset - nx;
                int idx_bottom = offset + nx;

                float top, bottom, left, right;
                top = (y > 0) ? d_old[idx_top] : 0;
                bottom = (y < ny - 1) ? d_old[idx_bottom] : 0;
                left = (x > 0) ? d_old[idx_left] : 0;
                right = (x < nx - 1) ? d_old[idx_right] : 0;

                d_new[offset] = d_old[offset] + diffusion_number * (top + bottom + left + right - d_old[offset] * 4.0f);
                if (d_new[offset] < 0.0f)
                {
                    d_new[offset] = 0.0f;
                }
            }
        }

        __global__ void ripple(float *grid, int N, int tick)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int offset = x + y * blockDim.x * gridDim.x;

            if (x < N && y < N)
            {
                float fx = x - N / 2;
                float fy = y - N / 2;
                float d = sqrtf(fx * fx + fy * fy);
                grid[offset] = 128.0f + 127.0f * cos(d / 10.0f - tick / 7.0f) / (d / 10.0f + 1.0f);
            }
        }
    }

    void naive_mandelbrot(int *grid, int N, int max_iter, float x_center, float y_center, float zoom)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::naive_mandelbrot<<<numBlocks, threadsPerBlock>>>(grid, N, max_iter, x_center, y_center, zoom);
    }

    void julia(int *grid, int N, int max_iter, float x_center, float y_center, float zoom)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::julia<<<numBlocks, threadsPerBlock>>>(grid, N, max_iter, x_center, y_center, zoom);
    }

    void naive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float diffusion_number)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::naive_diffusion_iteration<<<numBlocks, threadsPerBlock>>>(d_old, d_new, nx, ny, diffusion_number);
    }

    void _ripple(float *grid, int N, int tick)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
        device::ripple<<<numBlocks, threadsPerBlock>>>(grid, N, tick);
    }
}