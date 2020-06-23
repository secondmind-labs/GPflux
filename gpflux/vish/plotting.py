import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm


def spherical_grid(resolution=100):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack([x.flatten(), y.flatten(), z.flatten()]).T


def plot_spherical_function(f, resolution=100, rescale_radius=False, ax=None):
    """
    f is a function which takes a N x 3 matrix of points on the sphere in
    cartesian coordinates, and returns a N, vector.
    Here we construc the cartesian coordinates in a big grid and then plot
    """

    grid = spherical_grid(resolution)
    fgrid = f(grid).reshape(resolution, resolution)

    # Set the aspect ratio to 1 so our sphere looks spherical
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if rescale_radius:
        scale = np.abs(fgrid)
    else:
        scale = 1.0

    # scale the colors
    fmax, fmin = fgrid.max(), fgrid.min()
    fcolors = (fgrid - fmin) / (fmax - fmin)

    ax.plot_surface(
        grid[:, 0].reshape(resolution, resolution) * scale,
        grid[:, 1].reshape(resolution, resolution) * scale,
        grid[:, 2].reshape(resolution, resolution) * scale,
        rstride=1,
        cstride=1,
        facecolors=cm.viridis(fcolors),
    )

    # Turn off the axis planes
    ax.set_axis_off()
    return ax
