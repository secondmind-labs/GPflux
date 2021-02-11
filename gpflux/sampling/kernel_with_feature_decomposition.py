#  Copyright (C) PROWLER.io 2020 - All Rights Reserved
#  Unauthorised copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
from typing import Optional, Union

import tensorflow as tf

import gpflow
from gpflow.base import TensorType

NoneType = type(None)


class ApproximateKernel(gpflow.kernels.Kernel):
    r"""
    This class encapsulates a kernel with feature functions \phi_i(x) and coefficients \lambda_i, such that
    k(x, x') \approx \sum_i \lambda_i \phi_i(x) \phi_i(x') [e.g. Mercer or Bochner decomposition].
    Feature-coefficient pairs could be e.g. eigenfunction-eigenvalue pairs (Mercer) or
    Fourier features with constant coefficients (Bochner).
    """

    def __init__(
        self, feature_functions: tf.keras.layers.Layer, feature_coefficients: TensorType,
    ):
        """
        :param feature_functions: has a __call__ that computes L features for any
            given inputs. For X [N, D] feature_functions(X) returns a tensor of shape [N, L]
        :param feature_coefficients: a list of L coefficients associated with the features.
        """
        self._feature_functions = feature_functions
        self._feature_coefficients = feature_coefficients  # [L, 1]

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        """Approximates the true kernel by an inner product between feature functions"""
        phi = self._feature_functions(X)  # [N, L]
        if X2 is None:
            phi2 = phi
        else:
            phi2 = self._feature_functions(X2)  # [N2, L]

        r = tf.matmul(
            phi, tf.transpose(self._feature_coefficients) * phi2, transpose_b=True
        )  # [N, N2]

        N1, N2 = tf.shape(phi)[0], tf.shape(phi2)[0]
        tf.debugging.assert_equal(tf.shape(r), [N1, N2])
        return r

    def K_diag(self, X: TensorType) -> TensorType:
        """Approximates the true kernel by an inner product between feature functions"""
        phi_squared = self._feature_functions(X) ** 2  # [N, L]
        r = tf.reduce_sum(phi_squared * tf.transpose(self._feature_coefficients), axis=1)  # [N,]
        N = tf.shape(X)[0]
        tf.debugging.assert_equal(tf.shape(r), [N])  # noqa: E231
        return r


class KernelWithFeatureDecomposition(gpflow.kernels.Kernel):
    r"""
    Encapsulates a kernel with feature functions \phi_i(x) and coefficients \lambda_i, such that
    k(x, x') \approx \sum_i \lambda_i \phi_i(x) \phi_i(x') [e.g. Mercer or Bochner decomposition].
    Feature-coefficient pairs could be e.g. eigenfunction-eigenvalue pairs (Mercer) or
    Fourier features with constant coefficients (Bochner).

    In certain cases, the analytical expression for the kernel is not available.
    Passing `None` in that case is allowed and `K` and `K_diag` will be computed
    using the approximation provided by the feature decomposition.
    """

    def __init__(
        self,
        kernel: Union[gpflow.kernels.Kernel, NoneType],
        feature_functions: tf.keras.layers.Layer,
        feature_coefficients: TensorType,
    ):
        """
        :param kernel: can be `None`, in that case there is no analytical expression
            associated with the infinite sum and we approximate the kernel based on
            the feature decomposition.
        :param feature_functions: has a __call__ that computes L features for any
            given inputs. For X [N, D] feature_functions(X) returns a tensor of shape [N, L]
        :param feature_coefficients: a list of L coefficients associated with the features.
        """
        super().__init__()

        if kernel is None:
            self._kernel = ApproximateKernel(feature_functions, feature_coefficients)
        else:
            self._kernel = kernel

        self._feature_functions = feature_functions
        self._feature_coefficients = feature_coefficients  # [L, 1]
        tf.ensure_shape(self._feature_coefficients, tf.TensorShape([None, 1]))

    @property
    def feature_functions(self) -> tf.keras.layers.Layer:
        return self._feature_functions

    @property
    def feature_coefficients(self) -> TensorType:
        return self._feature_coefficients

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        return self._kernel.K(X, X2)

    def K_diag(self, X: TensorType) -> TensorType:
        return self._kernel.K_diag(X)
