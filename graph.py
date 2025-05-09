import numpy as np
import matplotlib.pyplot as plt

grid = np.loadtxt("build/data.csv", delimiter=",")
plt.imshow(grid, cmap="viridis")
plt.colorbar()
plt.savefig("grid_image.png", dpi=300, bbox_inches='tight')