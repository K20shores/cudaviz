import matplotlib.pyplot as plt

from cudaviz.mandelbrot import mandelbrot

grid = mandelbrot(N=1000, max_iter=2000)

plt.imshow(grid, cmap="viridis")
plt.colorbar()
plt.savefig("grid_image.png", dpi=300, bbox_inches='tight')