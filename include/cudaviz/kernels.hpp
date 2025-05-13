#pragma once

namespace cudaviz
{
    namespace kernels {
        void naive_mandelbrot(int *grid, int N, int max_iter, float x_center, float y_center, float zoom);
        void julia(int *grid, int N, int max_iter, float x_center, float y_center, float zoom);
        void naive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float diffusion_number);
        void ripple(float *grid, int N, int tick);
        void ray_trace(unsigned char* data, int N, int n_spheres);
    }
}