#pragma once

namespace cudaviz
{
    void naive_mandelbrot(int *grid, int N, int max_iter = 100, float x_center = 0.0f, float y_center = 0.0f, float zoom = 1.0f);
    void julia(int *grid, int N, int max_iter = 100, float x_center = 0.0f, float y_center = 0.0f, float zoom = 1.0f);
    void naive_diffusion_iteration(float *d_old, float *d_new, int nx, int ny, float diffusion_number);
}