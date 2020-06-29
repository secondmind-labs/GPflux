import numpy as np
import pytest
import tensorflow as tf
from scipy.integrate import quad
from scipy.special import gamma
from scipy.special import gegenbauer as scipy_gegenbauer

from gpflux.vish.spherical_harmonics import (
    FastSphericalHarmonicsCollection,
    SphericalHarmonicsCollection,
    SphericalHarmonicsLevel,
)
from gpflux.vish.misc import spherical_to_cartesian, spherical_to_cartesian_4d


@pytest.mark.parametrize("max_degree", range(1, 10, 3))
def test_orthonormal_basis_3d(max_degree):
    """Numerical check that int_{S^2} Y_i(x) Y_j(x) dx = dirac(i,j)"""

    num_grid = 300
    dimension = 3

    theta = np.linspace(0, 2 * np.pi, num_grid)  # [N]
    phi = np.linspace(0, np.pi, num_grid)  # [N]
    theta, phi = np.meshgrid(theta, phi)  # [N, N], [N, N]
    x_spherical = np.c_[theta.reshape(-1, 1), phi.reshape(-1, 1)]  # [N^2, 2]
    x_cart = spherical_to_cartesian(x_spherical)

    harmonics = FastSphericalHarmonicsCollection(dimension, max_degree)
    harmonics_at_x = harmonics(x_cart).numpy()  # [M, N^2]

    d_x_spherical = 2 * np.pi ** 2 / num_grid ** 2
    inner_products = (
        harmonics_at_x
        # sin(phi) to account for the surface area element of S^2
        @ (harmonics_at_x.T * np.sin(x_spherical[:, [1]]))
        * d_x_spherical
    )

    np.testing.assert_array_almost_equal(
        inner_products, np.eye(len(harmonics_at_x)), decimal=2
    )


@pytest.mark.parametrize("max_degree", range(1, 8, 3))
def test_orthonormal_basis_4d(max_degree):
    """Numerical check that int_{S^3} Y_i(x) Y_j(x) dx = dirac(i,j)"""
    num_grid = 25
    dimension = 4

    theta1 = np.linspace(0, 2 * np.pi, num_grid)
    theta2 = np.linspace(0, np.pi, num_grid)
    theta3 = np.linspace(0, np.pi, num_grid)
    theta1, theta2, theta3 = np.meshgrid(theta1, theta2, theta3)
    x_spherical = np.c_[
        theta1.reshape((-1, 1)),
        theta2.reshape((-1, 1)),
        theta3.reshape((-1, 1)),
    ]  # [N^3, 3]
    x_cart = spherical_to_cartesian_4d(x_spherical)

    harmonics = FastSphericalHarmonicsCollection(dimension, max_degree)
    harmonics_at_x = harmonics(x_cart).numpy()  # [M, N^3]

    d_x_spherical = 2 * np.pi ** 3 / num_grid ** 3

    inner_products = (
        np.ones((len(harmonics_at_x), len(harmonics_at_x))) * np.nan
    )

    for i, Y1 in enumerate(harmonics_at_x):
        for j, Y2 in enumerate(harmonics_at_x):
            v = np.sum(
                Y1
                * Y2
                # account for surface area element of S^3 sphere
                * np.sin(x_spherical[:, -1]) ** 2
                * np.sin(x_spherical[:, -2])
                * d_x_spherical
            )
            inner_products[i, j] = v

    np.testing.assert_array_almost_equal(
        inner_products, np.eye(len(harmonics)), decimal=1
    )


@pytest.mark.parametrize("dimension", range(3, 10, 3))
@pytest.mark.parametrize("max_degree", range(2, 7, 3))
def test_equality_spherical_harmonics_collections(dimension, max_degree):

    fast_harmonics = FastSphericalHarmonicsCollection(dimension, max_degree)
    harmonics = SphericalHarmonicsCollection(dimension, max_degree)

    num_points = 100
    X = np.random.randn(100, dimension)
    # make unit vectors
    X /= np.sum(X ** 2, axis=-1, keepdims=True) ** 0.5

    np.testing.assert_array_almost_equal(
        fast_harmonics(X).numpy(), harmonics(X).numpy(),
    )


@pytest.mark.parametrize("dimension", range(3, 7, 1))
@pytest.mark.parametrize("degree", range(1, 7, 3))
def test_addition_theorem(dimension, degree):
    harmonics = SphericalHarmonicsLevel(dimension, degree)
    X = np.random.randn(100, dimension)
    X = X / (np.sum(X ** 2, keepdims=True, axis=1) ** 0.5)
    harmonics_at_X = harmonics(X)[..., None]  # [M:=N(dimension, degree), N, 1]
    harmonics_xxT = tf.matmul(
        harmonics_at_X, harmonics_at_X, transpose_b=True
    )  # [M, N, N]

    # sum over all harmonics in the level
    # addition_manual = harmonics_at_X.T @ harmonics_at_X  # [N, N]
    addition_manual = tf.reduce_sum(harmonics_xxT, axis=0).numpy()  # [N, N]
    addition_theorem = harmonics.addition(X).numpy()

    np.testing.assert_array_almost_equal(addition_manual, addition_theorem)

    np.testing.assert_array_almost_equal(
        np.diag(addition_manual)[..., None], harmonics.addition_at_1(X).numpy()
    )
