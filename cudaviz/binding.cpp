#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cudaviz/Mandelbrot>

namespace py = pybind11;

PYBIND11_MODULE(_cudaviz, m)
{
  m.def("_mandelbrot", &cudaviz::mandelbrot, "Iterate an NxN grid to form the Mandelbrot set");
}