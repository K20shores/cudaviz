#include <cudaviz/Mandelbrot>
#include <iostream>

namespace cudaviz
{
    namespace device
    {

        __device__ float scale(int k, int N, float min, float max)
        {
            return min + k * ((max - min) / N);
        }

        __global__ void mandelbrot_iteration(int *grid, int N, int max_iter, float xcenter, float y_center, float zoom)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            float scaled_x_width = 2.47 * zoom;
            float scaled_y_height = 2.24 * zoom;
            double x = 0;
            double y = 0;
            double xnew = 0;
            double ynew = 0;

            double x0 = scale(i, N, xcenter - scaled_x_width / 2, xcenter + scaled_x_width / 2);
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

        __global__ void naiive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float alpha)
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

                d_new[offset] = d_old[offset] + alpha * (top + bottom + left + right - d_old[offset] * 4.0f);
                if (d_new[offset] < 0.0f)
                {
                    d_new[offset] = 0.0f;
                }
            }
        }
    }

    void mandelbrot_iteration(int *grid, int N, int max_iter, float xcenter, float y_center, float zoom)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::mandelbrot_iteration<<<numBlocks, threadsPerBlock>>>(grid, N, max_iter, xcenter, y_center, zoom);
        cudaDeviceSynchronize();
    }

    void naiive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float alpha)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
        device::naiive_diffusion_iteration<<<numBlocks, threadsPerBlock>>>(d_old, d_new, nx, ny, alpha);
        cudaDeviceSynchronize();
    }
}