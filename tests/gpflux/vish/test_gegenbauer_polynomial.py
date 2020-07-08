import operator
from functools import reduce

import numpy as np
import pytest
import tensorflow as tf
from scipy.integrate import quad
from scipy.special import gegenbauer as scipy_gegenbauer
from scipy.special import factorial, gamma

from gpflux.vish.misc import surface_area_sphere
from gpflux.vish.gegenbauer_polynomial import (
    Polynomial,
    Gegenbauer,
)


def test_polynomial():
    x = np.random.rand(5, 6)
    cs = np.array([2.0, 4.0, 3.5], dtype=np.float64)
    ps = np.array([3.0, 5.0, 8.0], dtype=np.float64)

    np.testing.assert_array_almost_equal(
        reduce(operator.add, (c * x ** p for c, p in zip(cs, ps))),
        Polynomial(cs, ps)(x).numpy(),
    )

    np.testing.assert_array_almost_equal(
        cs[0] * x ** ps[0] + cs[1] * x ** ps[1] + cs[2] * x ** ps[2],
        Polynomial(cs, ps)(x).numpy(),
    )


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 9.0])
@pytest.mark.parametrize("n", range(10))
def test_Gegenbauer(alpha, n):
    x = np.linspace(-1, 1, 1_000).reshape(-1, 1)

    np.testing.assert_array_almost_equal(
        scipy_gegenbauer(n, alpha)(x),
        Gegenbauer(n, alpha)(tf.convert_to_tensor(x)).numpy(),
    )


@pytest.mark.parametrize("max_degree,dimension", [(8, 10), (7, 20), (6, 31)])
def test_Gegenbauer_extreme(max_degree, dimension):
    alpha = (dimension - 2) / 2
    x = np.linspace(-1, 1, 1_000).reshape(-1, 1)

    np.testing.assert_array_almost_equal(
        scipy_gegenbauer(max_degree, alpha)(x),
        Gegenbauer(max_degree, alpha)(tf.convert_to_tensor(x)).numpy(),
    )


@pytest.mark.parametrize("degree", range(8))
@pytest.mark.parametrize("dimension", range(3, 9))
def test_normalisation_gegenbauer(degree, dimension):
    omega_d = surface_area_sphere(dimension) / surface_area_sphere(dimension - 1)
    alpha = (dimension - 2) / 2
    gegenbauer = Gegenbauer(degree, alpha)

    def c(t):
        return gegenbauer(tf.cast(t, dtype=tf.float64)).numpy()

    def func(t):
        return c(t) ** 2 * (1 - t ** 2) ** (alpha - 0.5)

    desired = quad(func, -1, 1)[0]
    value = c(1) * alpha / (degree + alpha) * omega_d
    np.testing.assert_almost_equal(desired, value, decimal=5)

    # Definition Wiki https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    value2 = np.pi * 2 ** (1 - 2 * alpha) * gamma(degree + 2 * alpha)
    value2 /= factorial(degree) * (degree + alpha) * gamma(alpha) ** 2
    np.testing.assert_almost_equal(desired, value2, decimal=5)


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 9.0])
@pytest.mark.parametrize("n", range(10))
def test_gegenbauer_at_1(n, alpha):
    c_1 = Gegenbauer(n, alpha)(tf.cast(1.0, dtype=tf.float64)).numpy()
    expected = gamma(2 * alpha + n) / gamma(2 * alpha) / factorial(n)
    np.testing.assert_almost_equal(c_1, expected)
