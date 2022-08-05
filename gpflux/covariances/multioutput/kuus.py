import tensorflow as tf
from typing import Union
from gpflow.inducing_variables import FallbackSharedIndependentInducingVariables

from gpflow.base import TensorLike


from gpflow.kernels import (
    SharedIndependent,
)


from gpflux.covariances.dispatch import Kuu


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def Kuu_shared_shared(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat
