
from _cudaviz import _ray_trace
from _cudaviz import RGB

def ray_trace(N: int = 10, n_spheres: int = 10) -> list[list]:
  """
  Create an image on an N x N grid of ray traced spheres that are randomly generated

  Args:
    N (int): The number of points in the x and y direction
    n_spheres (int): The number of spheres to randomly generate

  Returns:
    list[list]: A 2D list of RGB types representing the ray traced image
  """

  return _ray_trace(N, n_spheres)