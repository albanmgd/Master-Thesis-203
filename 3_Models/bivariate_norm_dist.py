import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Define the parameters of the bivariate normal distribution
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create a grid of points for the x and y axes
x, y = np.mgrid[-3:3:.05, -3:3:.05]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Create a figure with a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate the density function for each point on the grid
z = np.zeros(x.shape)
for i in range(len(x)):
    for j in range(len(y)):
        z[i, j] = multivariate_normal.pdf(pos[i, j], mean=mean, cov=cov)

# Plot the density function as a surface in 3D
ax.plot_surface(x, y, z, cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')
plt.show()
