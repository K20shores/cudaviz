#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cudaviz/Mandelbrot>

namespace py = pybind11;

PYBIND11_MODULE(_cudaviz, m)
{
  m.def("_mandelbrot", &cudaviz::mandelbrot,
        py::arg("max_iter") = cudaviz::DEFAULT_MAX_ITER,
        py::arg("N") = cudaviz::DEFAULT_N,
        py::arg("x_center") = cudaviz::DEFAULT_X_CENTER,
        py::arg("y_center") = cudaviz::DEFAULT_Y_CENTER,
        py::arg("zoom") = cudaviz::DEFAULT_ZOOM,
        "Iterate an NxN grid to form the Mandelbrot set with adjustable center and zoom");
}