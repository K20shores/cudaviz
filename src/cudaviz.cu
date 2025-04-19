#include <cudaviz/cudzviz.cuh>

namespace cudaviz {
    inline namespace device {
        __global__ void addOne(int *data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            data[idx] += 1;
        }
    }

    void addOneDriver(int *data) {
        int blockSize = 256;
        int numBlocks = (1024 + blockSize - 1) / blockSize;
        device::addOne<<<numBlocks, blockSize>>>(data);
        cudaDeviceSynchronize();
    }
}