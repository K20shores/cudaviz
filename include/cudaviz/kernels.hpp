#pragma once

namespace cudaviz
{
    void naive_mandelbrot(int *grid, int N, int max_iter, float x_center, float y_center, float zoom);
    void julia(int *grid, int N, int max_iter, float x_center, float y_center, float zoom);
    void naive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float diffusion_number);
    void _ripple(float *grid, int N, int tick);
}