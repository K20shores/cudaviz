#pragma once

#include <cudaviz/cudzviz.cuh>

namespace cudaviz {
    void addOne(int *data) {
        addOneDriver(data);
    }
}