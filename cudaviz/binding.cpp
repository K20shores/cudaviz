#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cudaviz/cudaviz>
#include <format>

namespace py = pybind11;

PYBIND11_MODULE(_cudaviz, m)
{
      m.def("_naive_mandelbrot", &cudaviz::naive_mandelbrot,
            py::arg("max_iter") = cudaviz::DEFAULT_MAX_ITER,
            py::arg("N") = cudaviz::DEFAULT_N,
            py::arg("x_center") = cudaviz::DEFAULT_X_CENTER,
            py::arg("y_center") = cudaviz::DEFAULT_Y_CENTER,
            py::arg("zoom") = cudaviz::DEFAULT_ZOOM,
            "Iterate an NxN grid to form the Mandelbrot set with adjustable center and zoom");

      m.def("_julia", &cudaviz::julia,
            py::arg("max_iter") = cudaviz::DEFAULT_MAX_ITER,
            py::arg("N") = cudaviz::DEFAULT_N,
            py::arg("x_center") = cudaviz::DEFAULT_X_CENTER,
            py::arg("y_center") = cudaviz::DEFAULT_Y_CENTER,
            py::arg("zoom") = cudaviz::DEFAULT_ZOOM,
            "Iterate an NxN grid to form the Mandelbrot set with adjustable center and zoom using complex math");

      m.def("_naive_diffusion", &cudaviz::naive_diffusion,
            py::arg("nx") = cudaviz::DEFAULT_NX,
            py::arg("ny") = cudaviz::DEFAULT_NY,
            py::arg("nt") = cudaviz::DEFAULT_NT,
            py::arg("dt") = cudaviz::DEFAULT_DT,
            py::arg("alpha") = cudaviz::DEFAULT_ALPHA,
            py::arg("central_temperature") = cudaviz::DEFAULT_CENTRAL_TEMPERATURE,
            py::arg("spread") = cudaviz::DEFAULT_SPREAD,
            "Perform diffusion on a grid");

      m.def("_ripple", &cudaviz::ripple,
            py::arg("N") = cudaviz::DEFAULT_DIM,
            py::arg("tick") = cudaviz::DEFAULT_TICK,
            "Create a ripple image");

      py::class_<cudaviz::RGB>(m, "RGB")
          .def(py::init<>())
          .def_readwrite("r", &cudaviz::RGB::r)
          .def_readwrite("g", &cudaviz::RGB::g)
          .def_readwrite("b", &cudaviz::RGB::b)
          .def("__str__", [](const cudaviz::RGB &p)
               { return "RGB"; })
          .def("__repr__", [](const cudaviz::RGB &p)
               { return std::format("<RGB: (r: {}, g: {}, b: {})>", p.r, p.g, p.b); });
      ;

      m.def("_ray_trace", &cudaviz::ray_trace,
            py::arg("N") = cudaviz::DEFAULT_RAY_DIM,
            "Create a ray-traced image from randomly generated spheres");

      m.def("_matmul", &cudaviz::matmul,
            py::arg("N") = cudaviz::DEFAULT_MATRIX_SIZE,
            "Multiply an NxN matrix together and return the time this takes");

      m.def("_tiled_matmul", &cudaviz::tiled_matmul,
            py::arg("N") = cudaviz::DEFAULT_MATRIX_SIZE,
            "Multiply an NxN matrix together and return the time this takes");
}