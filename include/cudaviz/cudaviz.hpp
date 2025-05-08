#pragma once

namespace cudaviz
{
    void setIndex(int *data);
    void add(float *A, float *B, float *C, int N);
    void matAdd(float *A, float *B, float *C, int N);
    void saxpy(float a, float* x, float* y, int N);
}