
from _cudaviz import _ray_trace
from _cudaviz import RGB

def ray_trace(N: int = 1024) -> list[list]:
  """
  Create an image on an N x N grid of ray traced spheres that are randomly generated

  Args:
    N (int): The number of points in the x and y direction

  Returns:
    list[list]: A 2D list of RGB types representing the ray traced image
  """

  return _ray_trace(N)