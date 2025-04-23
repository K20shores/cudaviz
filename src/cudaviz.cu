#include <cudaviz/cudaviz.hpp>

namespace cudaviz {
    namespace device {
        __global__ void addOne(int *data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            data[idx] = idx;
        }
    }

    void addOne(int *data) {
        device::addOne<<<4, 4>>>(data);
        cudaDeviceSynchronize();
    }
}