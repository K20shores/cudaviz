from _cudaviz import _matmul

def matmul(N: int = 1024):
  """ 
  Multiply two NxN matrices together and report the time it takes

  Args:
    N (int, optional): The size of the grid (N x N). Defaults to 1024.

  Returns:
    float: the time it took to multiple
  """
  return _matmul(N)