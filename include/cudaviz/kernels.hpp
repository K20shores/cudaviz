#pragma once

namespace cudaviz
{
    void mandelbrotIteration(int *grid, int N, int max_iter=100, float x_center=0.0f, float y_center=0.0f, float zoom=1.0f);
}