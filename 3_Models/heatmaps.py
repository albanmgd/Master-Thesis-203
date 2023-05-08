import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

# Illustrate covariance matrix and function
def exponentiated_quadratic(xa, xb, l=1):
    """Exponentiated quadratic kernel with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def periodic_kernel(xa, xb, p=1, length=1):
    """Periodic kernel with l=1, p=1"""
    sq_norm = (-0.5 * np.sin(np.pi * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') / p) ** 2) / (length ** 2)
    return np.exp(sq_norm)


# Illustrate covariance matrix and function

# Show covariance matrix example from exponentiated quadratic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7))
xlim = (-3, 3)
X = np.expand_dims(np.linspace(*xlim, 25), 1)
Σ = periodic_kernel(X, X)
# Plot covariance matrix
im = ax1.imshow(Σ, cmap=cm.YlGnBu)
cbar = plt.colorbar(
    im, ax=ax1, fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(xa,xb)$', fontsize=10)
ax1.set_title((
    'Covariance matrix\nfor the periodic kernel \n'
    ))
ax1.set_xlabel('xa', fontsize=13)
ax1.set_ylabel('xb', fontsize=13)
ticks = list(range(xlim[0], xlim[1] + 1))
ax1.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
ax1.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
ax1.set_xticklabels(ticks)
ax1.set_yticklabels(ticks)
ax1.grid(False)

# Show covariance with X=0
xlim = (-3, 3)
X = np.expand_dims(np.linspace(*xlim, 25), 1)
Σ_expo = exponentiated_quadratic(X, X)
# Plot covariance matrix
im2 = ax2.imshow(Σ_expo, cmap=cm.YlGnBu)
cbar2 = plt.colorbar(
    im2, ax=ax2, fraction=0.045, pad=0.05)
cbar2.ax.set_ylabel('$k(xa,xb)$', fontsize=10)
ax2.set_title((
    'Covariance matrix\nfor the R.B.F. kernel \n'
    ))
ax2.set_xlabel('xa', fontsize=13)
ax2.set_ylabel('xb', fontsize=13)
# ax2.set_ylim([0, 1.1])

ticks = list(range(xlim[0], xlim[1] + 1))
ax2.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
ax2.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
ax2.set_xticklabels(ticks)
ax2.set_yticklabels(ticks)
ax2.legend(loc=1)

fig.tight_layout()
plt.show()
#
