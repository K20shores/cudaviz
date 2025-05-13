from _cudaviz import _naive_diffusion

def naive_diffusion(nx: int = 100, ny: int = 100, nt: int = 100, dt: float = 0.1, alpha: float = 2, central_temperature: float = 100, spread: float = 10) -> list[list[list]]:
  """
  A naiive diffusion function to generate a 2D diffusion pattern. 
  
  This uses a simple update pattern which simply takes the average of the neighboring points.

  T_new = T_old + d * (T_left + T_right + T_up + T_down - 4 * T_old)

  Args:
    nx (int): Number of x points.
    ny (int): Number of y points.
    nt (int): Number of time steps.
    dt (float): Time step size.
    alpha (float): Diffusion coefficient.
    central_temperature (float): Initial temperature at the center of the grid.
    spread (float): Temperature spread around the center.

  Returns:
    list[list[list]]: A 3D list representing the diffusion pattern.
  """

  return _naive_diffusion(nx, ny, nt, alpha, dt, central_temperature, spread)