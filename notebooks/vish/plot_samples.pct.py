import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from gpflow.config import default_float

from gpflux.vish.kernels import Matern
from gpflux.vish.plotting import plot_spherical_function
from gpflux.vish.spherical_harmonics import (
    FastSphericalHarmonicsCollection,
    SphericalHarmonicsCollection,
    SphericalHarmonicsLevel,
    num_harmonics,
)
from gpflux.vish.gegenbauer_polynomial import Gegenbauer
from gpflux.vish.matern_spectral_density import spectral_density

from notebooks.vish.ci_utils import is_running_pytest

DIM = 3
NU = 0.5
L = 3 if is_running_pytest() else 20
VAR = 1.0
RES = 10 if is_running_pytest() else 100


kernel = Matern(DIM, L, NU)

def sample(x):
    N = tf.shape(x)[0]
    K = kernel.K(x) + tf.eye(N, dtype=default_float()) * 1e-6  # [N, N]
    f = tf.linalg.cholesky(K) @ tf.random.normal((N, 1), dtype=default_float())  # [N, 1]
    return f.numpy()

plot_spherical_function(sample, resolution=RES)
plt.show()
