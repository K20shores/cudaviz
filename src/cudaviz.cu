#include <cudaviz/kernels.hpp>
#include <cudaviz/RayTrace>

#include "check_error.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <format>

#include <cuda_fp16.h>
using half_t = __half;

#include <mma.h>

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)

namespace cudaviz
{
    constexpr int TILE_WIDTH = 32;

    // WMMA fragment dimensions
    constexpr int WMMA_M = 16; // Number rows in tiles of A and C
    constexpr int WMMA_N = 16; // Number cols in tiles of B and C
    constexpr int WMMA_K = 16; // Number cols in tiles of A or rows in tiles of B

    namespace device
    {
        struct Sphere
        {
            float r, g, b;
            float radius;
            float x, y, z;

            __device__ float hit(float ox, float oy, float *n)
            {
                float dx = ox - x;
                float dy = oy - y;
                if (dx * dx + dy * dy < radius * radius)
                {
                    float dz = sqrtf(radius * radius - (dx * dx + dy * dy));
                    *n = dz / radius;
                    return dz + z;
                }
                return -INF;
            }
        };

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

        __constant__ device::Sphere spheres[N_SPHERES];
        __global__ void ray_trace(unsigned char *data, int N)
        {
            int x = threadIdx.x + blockDim.x * blockIdx.x;
            int y = threadIdx.y + blockDim.y * blockIdx.y;
            int offset = x + y * blockDim.x * gridDim.x;
            float ox = (x - N / 2);
            float oy = (y - N / 2);

            if (x >= N || y >= N)
                return;

            // if (x == 0 && y == 0)
            // {

            //     for (int i = 0; i < N_SPHERES; ++i)
            //     {
            //         printf("(%f, %f, %f) color (%f, %f, %f), radius: %f\n",
            //             spheres[i].x,
            //             spheres[i].y,
            //             spheres[i].z,
            //             spheres[i].r,
            //             spheres[i].g,
            //             spheres[i].b,
            //             spheres[i].radius
            //         );
            //     }
            // }

            float r = 1, g = 1, b = 1;
            float maxz = -INF;
            for (int i = 0; i < N_SPHERES; ++i)
            {
                float fscale;
                float t = spheres[i].hit(ox, oy, &fscale);
                // printf("%i, %i: %f\n", x, y, t);
                if (t > maxz)
                {
                    r = spheres[i].r * fscale;
                    g = spheres[i].g * fscale;
                    b = spheres[i].b * fscale;
                    maxz = t;
                }
            }

            data[offset * 3 + 0] = (int)(r * 255);
            data[offset * 3 + 1] = (int)(g * 255);
            data[offset * 3 + 2] = (int)(b * 255);
        }

        __global__ void matmul(float *A, float *B, float *C, int N)
        {
            int i = threadIdx.y + blockDim.y * blockIdx.y;
            int j = threadIdx.x + blockDim.x * blockIdx.x;
            if (i < N && j < N)
            {
                int c_ij = i * N + j;
                float value = 0;
                for (int k = 0; k < N; ++k)
                {
                    int a_ik = i * N + k;
                    int b_kj = k * N + j;
                    value += A[a_ik] * B[b_kj];
                }
                C[c_ij] = value;
            }
        }

        __global__ void tiled_matmul(float *A, float *B, float *C, int N)
        {
            __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

            int i = TILE_WIDTH * blockIdx.y + threadIdx.y;
            int j = TILE_WIDTH * blockIdx.x + threadIdx.x;

            float value = 0;
            for (int phase = 0; phase < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++phase)
            {
                int k_col = phase * TILE_WIDTH + threadIdx.x;
                int k_row = phase * TILE_WIDTH + threadIdx.y;
                if ((i < N) && (k_col) < N)
                {
                    sh_A[threadIdx.y][threadIdx.x] = A[i * N + k_col];
                }
                else
                {
                    sh_A[threadIdx.y][threadIdx.x] = 0.0f;
                }
                if ((j < N) && (k_row) < N)
                {
                    sh_B[threadIdx.y][threadIdx.x] = B[j + N * (k_row)];
                }
                else
                {
                    sh_B[threadIdx.y][threadIdx.x] = 0.0f;
                }
                __syncthreads();

                for (int k = 0; k < TILE_WIDTH; ++k)
                {
                    value += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];
                }
                __syncthreads();
            }
            if ((i < N) && (j < N))
            {
                C[i * N + j] = value;
            }
        }

        // based off of https://0mean1sigma.com/tgemm/
        __global__ void tensor_matmul(half *A, half *B, float *C, int N)
        {
            int tile_m = blockIdx.y;
            int tile_n = blockIdx.x;

            int row = tile_m * WMMA_M;
            int col = tile_n * WMMA_N;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

            for (int k = 0; k < N; k += WMMA_K)
            {
                int a_off = row * N + k;
                int b_off = k * N + col;
                if (row < N && k < N && col < N)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag, A + a_off, N);
                    nvcuda::wmma::load_matrix_sync(b_frag, B + b_off, N);
                    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }

            // Write output tile
            if (row < N && col < N)
            {
                nvcuda::wmma::store_matrix_sync(C + row * N + col, c_frag, N, nvcuda::wmma::mem_row_major);
            }
        }
    }

    namespace kernels
    {
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

        void ripple(float *grid, int N, int tick)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
            device::ripple<<<numBlocks, threadsPerBlock>>>(grid, N, tick);
        }

        void ray_trace(unsigned char *data, int N)
        {
            std::vector<device::Sphere> spheres(N_SPHERES);

            for (int i = 0; i < N_SPHERES; ++i)
            {
                spheres[i].r = rnd(1.0f);
                spheres[i].g = rnd(1.0f);
                spheres[i].b = rnd(1.0f);
                spheres[i].x = rnd(1000.0f) - 500.0f;
                spheres[i].y = rnd(1000.0f) - 500.0f;
                spheres[i].z = rnd(1000.0f) - 500.0f;
                spheres[i].radius = rnd(100.0f) + 20.0f;
                // std::cout << std::format("Sphere at ({}, {}, {}) color ({}, {}, {}) radius {}\n",
                //                          spheres[i].x,
                //                          spheres[i].y,
                //                          spheres[i].z,
                //                          spheres[i].r,
                //                          spheres[i].g,
                //                          spheres[i].b,
                //                          spheres[i].radius);
            }

            CUDA_CHECK(cudaMemcpyToSymbol(device::spheres, spheres.data(), N_SPHERES * sizeof(device::Sphere)));

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
            device::ray_trace<<<numBlocks, threadsPerBlock>>>(data, N);
        }

        void matmul(float *A, float *B, float *C, int N)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
            device::matmul<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        }

        void tiled_matmul(float *A, float *B, float *C, int N)
        {
            dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
            dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
            device::tiled_matmul<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
        }

        void tensor_matmul(half_t *A, half_t *B, float *C, int N)
        {
            dim3 threadsPerBlock(32, 1, 1); // one warp per block
            dim3 numBlocks((N + WMMA_N - 1) / WMMA_N,
                           (N + WMMA_M - 1) / WMMA_M);
            device::tensor_matmul<<<gridDim, blockDim, sharedMemBytes>>>(A, B, C, N);
        }
    }
}