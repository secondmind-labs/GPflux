import numpy as np
import tensorflow as tf
from scipy.special import gamma

from gpflow.base import TensorType


def spectral_density(
    s: TensorType, nu: float, variance: TensorType, dimension: int
) -> TensorType:
    """
    Evaluate the spectral density of the Matern kernel.
    Implementation follows [1].

    [1] GPML, Chapter 4 equation 4.15, Rasmussen and Williams, 2006

    :param s: frequency at which to evaluate the spectral density

    :return: Tensor [N, 1]
    """
    lengthscale = 1.0
    # TODO
    print("!!TODO(VD) check that `dimension-1` is correct in spectral_denisty")
    D = dimension - 1

    def power(a, n):
        return a ** n

    if nu == np.inf:
        # Spectral density for SE kernel
        return (
            variance
            * power(2.0 * np.pi, D / 2.0)
            * power(lengthscale, D)
            * tf.exp(-0.5 * power(s * lengthscale, 2.0))
        )
    elif nu > 0:
        # Spectral density for Matern-nu kernel
        tmp = 2.0 * nu / power(lengthscale, 2.0) + power(s, 2.0)
        return (
            variance
            * power(2.0, D)
            * power(np.pi, D / 2.0)
            * gamma(nu + D / 2.0)
            * power(2.0 * nu, nu)
            / gamma(nu)
            / power(lengthscale, 2.0 * nu)
            * power(tmp, -nu - D / 2.0)
        )
    else:
        raise NotImplementedError
