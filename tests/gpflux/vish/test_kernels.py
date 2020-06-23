import itertools

import pytest
import numpy as np
import tensorflow as tf
from scipy.special import gegenbauer

import gpflow
from gpflux.vish.kernels import Matern, ArcCosine, Parameterised
from gpflux.vish.misc import chain
from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection, num_harmonics


@pytest.mark.parametrize(
    "kernel_class",
    [
        lambda dim, max_level: Parameterised(dim, max_level, variance=0.33),
        lambda dim, max_level: Matern(dim, max_level, nu=0.5),
        lambda dim, max_level: Matern(dim, max_level, nu=1.5),
        lambda dim, max_level: Matern(dim, max_level, nu=2.5),
    ],
)
@pytest.mark.parametrize("truncation_level", [5, 6])
@pytest.mark.parametrize("dimension", [3, 5])
def test_kernels(kernel_class, truncation_level, dimension):
    r"""
    Test that k(x, x') = \sum_l \sum_i \lambda_l \phi_{l, k}(x) \phi_{l, k}(x')
    """
    kernel = kernel_class(dimension, truncation_level)
    number_of_harmonics_per_level = [
        num_harmonics(dimension, degree) for degree in kernel.degrees
    ]
    Lambda_diag = chain(kernel.eigenvalues, number_of_harmonics_per_level)
    Lambda_diag = np.array(Lambda_diag).reshape(-1, 1)  # [M, 1]
    harmonics = SphericalHarmonicsCollection(dimension, truncation_level)

    x = np.random.randn(100, dimension)
    x /= np.sum(x ** 2, axis=1, keepdims=True) ** 0.5  # [N, 1]

    harmonics_x = harmonics(x)[..., None]  # [M, N, 1]
    harmonics_xxT = tf.matmul(
        harmonics_x, harmonics_x, transpose_b=True
    )  # [M, N, N]
    expected = tf.reduce_sum(
        Lambda_diag[..., None] * harmonics_xxT, axis=0
    )  # [N, N]

    actual = kernel.K(x)

    np.testing.assert_array_almost_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize("dimension", [3, 5])
def test_arccosine(dimension):
    r"""
    Test that k(x, x') = \sum_l \sum_i \lambda_l \phi_{l, k}(x) \phi_{l, k}(x'),
    for k arccosine 
    """
    truncation_level = 25

    x = np.random.randn(10, dimension)
    x /= np.sum(x ** 2, axis=1, keepdims=True) ** 0.5  # [N, D]

    kernel = ArcCosine(dimension)

    def _K_mercer_theorem(X, X2):
        """
        This is a copy form kernels.py::ZonalKernel:K().
        For standard zonal kernels, of which we do not explicitly know
        the analytical form for of k(x, x') we use this method to compute
        the kernel.
        """
        inner_product = tf.matmul(X, X2, transpose_b=True)  # [N1, N2]
        cs = tf.stack(
            values=[c(inner_product) for c in kernel.gegenbauers], axis=0
        )  # [L, N1, N2]
        factors = (
            kernel.degrees / kernel.alpha + 1.0
        ) / kernel.surface_area_sphere  # [L, 1]
        sum_level_phi_X_phi_X2 = factors[..., None] * cs  # [L, N1, N2]
        # eigenvalues
        Lambda_level = kernel.eigenvalues[..., None]  # [L, 1, 1]
        return tf.reduce_sum(
            Lambda_level * sum_level_phi_X_phi_X2, axis=0
        )  # [N1, N2]

    approx = _K_mercer_theorem(x, x).numpy()  # [N, N]
    exact = kernel.K(x, x).numpy()

    np.testing.assert_array_almost_equal(approx, exact, decimal=3)
