import timeit
import numpy as np
import tensorflow as tf

from gpflux.vish.plotting import plot_spherical_function
from gpflux.vish.spherical_harmonics import (
    FastSphericalHarmonicsCollection,
    SphericalHarmonicsCollection,
    SphericalHarmonicsLevel,
    num_harmonics,
)

from notebooks.ci_utils import is_running_pytest


if __name__ == "__main__":
    dimension = 3
    max_degree = 4

    fast_harmonics = FastSphericalHarmonicsCollection(
        dimension, max_degree, debug=False
    )
    harmonics = SphericalHarmonicsCollection(dimension, max_degree, debug=False)
    print("Number", len(fast_harmonics))
    print("Number", len(harmonics))

    num_points = 10 if is_running_pytest() else 100

    X = np.random.randn(100, dimension)
    X /= np.sum(X ** 2, axis=-1, keepdims=True) ** 0.5

    print("First")
    print(timeit.timeit(lambda: fast_harmonics(X), number=1))
    print(timeit.timeit(lambda: harmonics(X), number=1))

    print("Second")
    print(timeit.timeit(lambda: fast_harmonics(X), number=100) / 100)
    print(timeit.timeit(lambda: harmonics(X), number=100) / 100)
