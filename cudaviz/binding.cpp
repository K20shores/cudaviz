#include <pybind11/pybind11.h>

#include <cudaviz/Mandelbrot>

namespace py = pybind11;

PYBIND11_MODULE(_cudaviz, m)
{
  m.def("mandelbrot", &mandelbrot, "Iterate an NxN grid to form the Mandelbrot set");
}