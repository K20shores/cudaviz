#include <cudaviz/Diffusion>
#include <cudaviz/kernels.hpp>

#include "check_error.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <math.h>

#include <cuda_runtime.h>

namespace cudaviz
{
  void set_initial_conditions(std::vector<float> &h_old, int nx, int ny, float central_temperature, float spread) {
    for(int y = 0; y < ny; ++y) {
      for(int x = 0; x < nx; ++x) {
        float X = x - nx / 2;
        float Y = y - ny / 2;
        h_old[y*nx + x] = central_temperature * exp(-(X * X + Y * Y) / spread);
      }
    }
  }

  std::vector<std::vector<std::vector<float>>> naive_diffusion(int nx, int ny, int nt, float D, float central_temperature, float spread)
  {
    std::size_t sz = nx * ny * sizeof(float);
    std::vector<float> h_old = std::vector<float>(nx * ny, 0.0f);

    set_initial_conditions(h_old, nx, ny, central_temperature, spread);

    float *d_old;
    float *d_new;

    CUDA_CHECK(cudaMalloc(&d_old, sz));
    CUDA_CHECK(cudaMalloc(&d_new, sz));

    CUDA_CHECK(cudaMemcpy(d_old, h_old.data(), sz, cudaMemcpyHostToDevice));

    std::vector<std::vector<std::vector<float>>> grid3D(nt, std::vector<std::vector<float>>(nx, std::vector<float>(ny, 0)));
    for(int y = 0; y < ny; ++y) {
      for(int x = 0; x < nx; ++x) {
        grid3D[0][y][x] = h_old[y*nx + x];
      }
    }

    // h = dx = dy 
    float h = 1.0f;
    float dt = 0.1f;
    float alpha = dt * D / (h * h);
    for (int t = 1; t < nt; ++t)
    {
      int num_substeps = static_cast<int>(1.0f / dt);
      for(int substep = 0; substep < num_substeps; ++substep) {
        float current_time = substep * dt;
        naiive_diffusion_iteration(d_old, d_new, nx, ny, alpha);
        std::swap(d_old, d_new);
      }

      CUDA_CHECK(cudaMemcpy(h_old.data(), d_old, sz, cudaMemcpyDeviceToHost));
      for(int y = 0; y < ny; ++y) {
        for(int x = 0; x < nx; ++x) {
          grid3D[t][y][x] = h_old[y*nx + x];
        }
      }
    }

    CUDA_CHECK(cudaFree(d_old));
    CUDA_CHECK(cudaFree(d_new));

    return grid3D;
  }
}