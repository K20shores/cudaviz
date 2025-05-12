#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cudaviz/Mandelbrot>
#include <cudaviz/Diffusion>

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

      m.def("_naive_diffusion", &cudaviz::naive_diffusion,
            py::arg("nx") = cudaviz::DEFAULT_NX,
            py::arg("ny") = cudaviz::DEFAULT_NY,
            py::arg("nt") = cudaviz::DEFAULT_NT,
            py::arg("dt") = cudaviz::DEFAULT_DT,
            py::arg("alpha") = cudaviz::DEFAULT_ALPHA,
            py::arg("central_temperature") = cudaviz::DEFAULT_CENTRAL_TEMPERATURE,
            py::arg("spread") = cudaviz::DEFAULT_SPREAD,
            "Perform diffusion on a grid");
}