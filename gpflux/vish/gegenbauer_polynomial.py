import math
from typing import List, Union, Tuple

import numpy as np
import tensorflow as tf
from scipy.special import loggamma
from scipy.special import gegenbauer as scipy_gegenbauer

from gpflow.base import TensorType


class Polynomial:
    r"""
    One-dimensional polynomial function expressed with coefficients and powers.
    The polynomial f(x) is given by f(x) = \sum_i c_i x^{p_i}, with x \in R^1.
    """

    def __init__(
        self, coefficients: Union[List, np.ndarray], powers: Union[List, np.ndarray],
    ):
        r"""
        The polynomial f(x) is given by f(x) = \sum_i c_i x^{p_i},
        with c coefficients and p powers. The number of coefficients and
        the number of powers must match.

        :param coefficients: list of weights
        :param powers: list of powers
        """
        assert len(coefficients) == len(powers)
        self.coefficients = coefficients
        self.powers = powers

    def __call__(self, x: TensorType) -> TensorType:
        """
        Evaluates the polynomial @ `x`
        :param x: 1D input values at which to evaluate the polynomial, [...]

        :return:
            function evaluations, same shape as `x` [...]
        """
        cs = tf.reshape(self.coefficients, (1, -1))  # [1, M]
        ps = tf.reshape(self.powers, (1, -1))  # [1, M]
        x_flat = tf.reshape(x, (-1, 1))  # [N, 1]
        val = tf.reduce_sum(cs * (x_flat ** ps), axis=1)  # [N, M]  # [N]
        return tf.reshape(val, tf.shape(x))


class Gegenbauer(Polynomial):
    r"""
    Gegenbauer polynomials or ultraspherical polynomials C_n^(α)(x)
    are orthogonal polynomials on the interval [−1,1] with respect
    to the weight function (1 − x^2)^{α–1/2} [1].

    [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials,
    [2] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """

    def __init__(self, n: int, alpha: float):
        """
        Gegenbauer polynomial C_n^(α)(z) of degree `n` and characterisation `alpha`.
        We represent the Gegenbauer function as a polynomial and compute its
        coefficients and corresponding powers.

        :param n: degree
        :param alpha: characterises the form of the polynomial.
            Typically changes with the dimension, alpha = (dimension - 2) / 2
        """

        coefficients, powers = self._compute_coefficients_and_powers(n, alpha)
        super().__init__(
            np.array(coefficients, dtype=np.float64),
            np.array(powers, dtype=np.float64),
        )
        self.n = n
        self.alpha = alpha
        self._at_1 = scipy_gegenbauer(self.n, self.alpha)(1.0)

    def _compute_coefficients_and_powers(
        self, n: int, alpha: float
    ) -> Tuple[List, List]:
        """
        Compute the weights (coefficients) and powers of the Gegenbauer functions
        expressed as polynomial.

        :param n: degree
        :param alpha:
        """
        coefficients, powers = [], []

        for k in range(math.floor(n / 2) + 1):  # k=0 .. floor(n/2) (incl.)
            # computes the coefficients in log space for numerical stability
            log_coef = loggamma(n - k + alpha)
            log_coef -= loggamma(alpha) + loggamma(k + 1) + loggamma(n - 2 * k + 1)
            log_coef += (n - 2 * k) * np.log(2)
            coef = np.exp(log_coef)
            coef *= (-1) ** k
            coefficients.append(coef)
            powers.append(n - 2 * k)

        return coefficients, powers

    def __call__(self, x: TensorType) -> TensorType:
        if self.n < 0:
            return tf.zeros_like(x)
        elif self.n == 0:
            return tf.ones_like(x)
        elif self.n == 1:
            return 2 * self.alpha * x
        else:
            return super().__call__(x)

    @property
    def value_at_1(self):
        """
        Gegenbauer evaluated at 1.0
        """
        return self._at_1
