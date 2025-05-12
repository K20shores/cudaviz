from _cudaviz import _naive_mandelbrot, _julia

def naive_mandelbrot(N: int = 100, max_iter: int = 1000, x_center: float = -0.75, y_center: float = 0.0, zoom: float = 1.0) -> list[list]:
  """
  Generate the Mandelbrot set on an N x N grid.

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 100.
    max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

  Returns:
    list[list]: A 2D list representing the Mandelbrot set.
  """

  return _naive_mandelbrot(max_iter, N, x_center, y_center, zoom)

def julia(N: int = 100, max_iter: int = 1000, x_center: float = -0.75, y_center: float = 0.0, zoom: float = 1.0) -> list[list]:
  """
  Generate the Mandelbrot set on an N x N grid.

  This function uses complex numbers to compute the Mandelbrot set.

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 100.
    max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

  Returns:
    list[list]: A 2D list representing the Mandelbrot set.
  """

  return _julia(max_iter, N, x_center, y_center, zoom)