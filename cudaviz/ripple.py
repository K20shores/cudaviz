from _cudaviz import _ripple

def ripple(N: int = 1024, tick: int = 0) -> list[list]:
  """
  Create a ripple image on an N x N grid at a specific tick;

  Args:
    N (int): The number of points in the x and y direction
    tick (int): The time tick

  Returns:
    list[list]: A 2D list representing the diffusion pattern.
  """

  return _ripple(N, tick)