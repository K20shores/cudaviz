#include <cudaviz/cudaviz.hpp>

namespace cudaviz {
    namespace device {
        __global__ void addOne(int *data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            data[idx] += 1;
        }
    }

    void addOne(int *data) {
        int blockSize = 256;
        int numBlocks = (1024 + blockSize - 1) / blockSize;
        device::addOne<<<numBlocks, blockSize>>>(data);
        cudaDeviceSynchronize();
    }
}