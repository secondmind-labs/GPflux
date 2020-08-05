#  Copyright (C) PROWLER.io 2020 - All Rights Reserved
#  Unauthorised copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
from typing import Union

import tensorflow as tf
import gpflow


from gpflow.base import TensorType


NoneType = type(None)


class ApproximateKernel(gpflow.kernels.Kernel):
    r"""
    Approximates a kernel with its eigenfunctions and eigenvalues such that
    k(x, x') \approx \sum_i \lambda_i \phi_i(x) \phi_i(x') [Mercer Decomposition],
    with \lambda_i the eigenvalues and \phi_i the eigenfunctions.
    For the sum going to infinity the approximation becomes exact.
    """

    def __init__(
        self, eigenfunctions: tf.keras.layers.Layer, eigenvalues: TensorType,
    ):
        """
        :param eigenfunctions: has a call that computes the first L eigenfeatures for any
            given inputs. For X [N, D] eigenfunctions(X) returns a tensor of shape [N, L]
        :param eigenvalues: a list of the first L eigenvalues associated with the eigenfeatures.
        """
        self._eigenfunctions = eigenfunctions
        self._eigenvalues = eigenvalues  # [L, 1]

    def K(self, X, X2=None):
        """Approximates the true kernel by an inner product between eigenfunctions"""
        phi = self._eigenfunctions(X)  # [N, L]
        if X2 is None:
            phi2 = phi
        else:
            phi2 = self._eigenfunctions(X2)  # [N2, L]

        r = tf.matmul(
            phi, tf.transpose(self._eigenvalues) * phi2, transpose_b=True
        )  # [N, N2]

        N1, N2 = tf.shape(phi)[0], tf.shape(phi2)[0]
        tf.debugging.assert_equal(tf.shape(r), [N1, N2])
        return r

    def K_diag(self, X):
        """Approximates the true kernel by an inner product between eigenfunctions"""
        phi_squared = self._eigenfunctions(X) ** 2  # [N, L]
        r = tf.reduce_sum(phi_squared * tf.transpose(self._eigenvalues), axis=1)  # [N,]
        N = tf.shape(X)[0]
        tf.debugging.assert_equal(tf.shape(r), [N,])  # noqa: E231
        return r


class KernelWithMercerDecomposition(gpflow.kernels.Kernel):
    r"""
    Encapsulates a kernel with its eigenfunctions and eigenvalues, such that
    k(x, x') \approx \sum_i \lambda_i \phi_i(x) \phi_i(x') [Mercer Decomposition],
    with \lambda_i the eigenvalues and \phi_i the eigenfunctions.

    In certain cases, the analytical expression for the kernel is not available.
    Passing `None` in that case is allowed and `K` and `K_diag` will be computed
    using the approximation provided by Mercer's decomposition.
    """

    def __init__(
        self,
        kernel: Union[gpflow.kernels.Kernel, NoneType],
        eigenfunctions: tf.keras.layers.Layer,
        eigenvalues: TensorType,
    ):
        """
        :param kernel: can be `None`, in that case there is no analytical expression
            associated with the infinite sum and we approximate the kernel based on
            Mercer's decomposition.
        :param eigenfunctions: has a __call__ that computes the first L eigenfeatures for any
            given inputs. For X [N, D] eigenfunctions(X) returns a tensor of shape [N, L]
        :param eigenvalues: a list of the first L eigenvalues associated with the eigenfeatures.
        """
        super().__init__()

        if kernel is None:
            self._kernel = ApproximateKernel(eigenfunctions, eigenvalues)
        else:
            self._kernel = kernel

        self._eigenfunctions = eigenfunctions
        self._eigenvalues = eigenvalues  # [L, 1]
        tf.ensure_shape(self._eigenvalues, tf.TensorShape([None, 1]))

    @property
    def eigenfunctions(self):
        return self._eigenfunctions

    @property
    def eigenvalues(self):
        return self._eigenvalues

    def K(self, X, X2=None):
        return self._kernel.K(X, X2)

    def K_diag(self, X):
        return self._kernel.K_diag(X)
