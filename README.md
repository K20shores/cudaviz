# cudaviz

This package contains a C library that is wrapped with pybind11 to expose functions in python
which are run on NVIDIA GPUs with CUDA.

The supported functions are

- Mandelbrot
- A naive 2D heat diffusion


## Build and Installation
On a platform with a working CUDA environment and CUDA-enabled NVIDIA GPU, the project should
be buildable without any configuration using cmake.

```
cmake -S . -B build
cmake --build build
```

There is an executable made `./build/cudaviz_example`, which right now is only used
for me to play around with random kernels quickly.

### Python

```
pip install -e .[dev]
```

From there, you can open up cudaviz and use it in python, like in either of the example scripts:

- [A mandebrot user with tkinter, supported over X11](/graph.py)
- [This notebook shows how to use the package and plt both the mandelbrot set and the heat diffuser](Plot.ipynb)