from _cudaviz import _mandelbrot

def mandelbrot(N: int = 100, max_iter: int = 1000, x_center: float = -0.75, y_center: float = 0.0, zoom: float = 1.0) -> list[list]:
  """
  Generate the Mandelbrot set on an N x N grid.

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 100.
    max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

  Returns:
    list[list]: A 2D list representing the Mandelbrot set.
  """

  return _mandelbrot(max_iter, N, x_center, y_center, zoom)