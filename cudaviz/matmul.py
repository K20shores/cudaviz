from _cudaviz import _matmul, _tiled_matmul

def matmul(N: int = 1024):
  """ 
  Multiply two NxN matrices together and report the time it takes

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 1024.

  Returns:
    float: the time it took to multiply
  """
  return _matmul(N)

def tiled_matmul(N: int = 1024):
  """ 
  Multiply two NxN matrices together and report the time it takes using a tiled algorithm

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 1024.

  Returns:
    float: the time it took to multiply
  """
  return _tiled_matmul(N)