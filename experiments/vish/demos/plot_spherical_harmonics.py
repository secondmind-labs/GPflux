import tensorflow as tf
import matplotlib.pyplot as plt

from gpflux.vish.plotting import plot_spherical_function
from gpflux.vish.spherical_harmonics import (
    FastSphericalHarmonicsCollection,
    SphericalHarmonicsCollection,
    SphericalHarmonicsLevel,
    num_harmonics,
)

from gpflow.ci_utils import is_continuous_integration as is_running_pytest


if __name__ == "__main__":
    dimension = 3
    degree = 2
    max_degree = 2
    resolution = 10 if is_running_pytest() else 100

    N = sum(num_harmonics(dimension, d) for d in range(max_degree))
    Ys = FastSphericalHarmonicsCollection(dimension, max_degree)

    for i in range(min(10, N)):
        plot_spherical_function(lambda x: Ys(x)[i].numpy(), resolution=resolution)
    plt.show()