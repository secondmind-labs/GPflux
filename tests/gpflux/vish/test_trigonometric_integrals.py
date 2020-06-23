import numpy as np
import pytest
import scipy.integrate as integrate

from gpflux.vish.trigonometric_integrals import (
    int_0_pi_cos_p,
    int_0_pi_sin_p,
    integrate_0_pi__J1_cos_k_sin_d,
    integrate_0_pi__pi_minus_t_sin_n_cos_m,
    integrate_0_upper__sin_n_cos_m,
)


@pytest.mark.parametrize("p", np.arange(10))
def test_int_0_pi_sin_p(p):
    def func(theta):
        return np.sin(theta) ** p

    int_quad = integrate.quad(func, 0, np.pi)[0]
    int_analytical = int_0_pi_sin_p(p)
    np.testing.assert_almost_equal(int_analytical, int_quad, decimal=6)


@pytest.mark.parametrize("p", np.arange(10))
def test_int_0_pi_cos_p(p):
    def func(theta):
        return np.cos(theta) ** p

    int_quad = integrate.quad(func, 0, np.pi)[0]
    int_analytical = int_0_pi_cos_p(p)
    np.testing.assert_almost_equal(int_analytical, int_quad, decimal=6)


@pytest.mark.parametrize("n", np.arange(10))
@pytest.mark.parametrize("m", np.arange(10))
@pytest.mark.parametrize("upper", ["half_pi", "pi"])
def test_integrate_0_upper__sin_n_cos_m(n, m, upper):
    def func(theta):
        return np.sin(theta) ** n * np.cos(theta) ** m

    up = np.pi if upper == "pi" else np.pi / 2
    desired = integrate.quad(func, 0, up)[0]
    actual = integrate_0_upper__sin_n_cos_m(n, m, upper)
    np.testing.assert_almost_equal(actual, desired, decimal=7)


@pytest.mark.parametrize("n", np.arange(10))
@pytest.mark.parametrize("m", np.arange(10))
def test_integrate_0_pi__pi_minus_t_sin_n_cos_m(n, m):
    def func(theta):
        return (np.pi - theta) * np.sin(theta) ** n * np.cos(theta) ** m

    desired = integrate.quad(func, 0, np.pi)[0]
    actual = integrate_0_pi__pi_minus_t_sin_n_cos_m(n, m)
    np.testing.assert_almost_equal(actual, desired, decimal=7)


@pytest.mark.parametrize("k", np.arange(2000)[::7])
def test_integrate_0_pi__J1_cos_k_sin_d(k):
    d = 1
    range_with_less_precision = range(300, 500)

    if k in range_with_less_precision:
        decimal = 6
    else:
        decimal = 7

    def kernel(theta):
        return np.sin(np.abs(theta)) + (np.pi - np.abs(theta)) * np.cos(theta)

    int_analytical = integrate_0_pi__J1_cos_k_sin_d

    def func(theta):
        return kernel(theta) * np.sin(theta) ** d * np.cos(theta) ** k

    desired = integrate.quad(func, 0, np.pi)[0]
    actual = int_analytical(k, d)
    np.testing.assert_almost_equal(actual, desired, decimal=decimal)
