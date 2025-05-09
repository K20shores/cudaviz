# cuda


```
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
cmake -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_CUDA_FLAGS="-I/usr/local/cuda/include" -DCMAKE_CXX_FLAGS="-I/usr/local/cuda/include"..
```