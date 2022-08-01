import tensorflow as tf
from typing import Union
from gpflow.inducing_variables import FallbackSharedIndependentInducingVariables

from gpflow.base import TensorLike

from gpflux.inducing_variables import (
    FallbackSharedIndependentDistributionalInducingVariables,
)
from gpflow.kernels import (
    SharedIndependent,
)
from gpflux.kernels import (
    DistributionalSharedIndependent
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

@Kuu.register(FallbackSharedIndependentDistributionalInducingVariables, DistributionalSharedIndependent)
def Kuu_distributional_shared_shared(
    inducing_variable: FallbackSharedIndependentDistributionalInducingVariables,
    kernel: DistributionalSharedIndependent,
    *,
    jitter: float = 0.0,
    seed: int = None
) -> tf.Tensor:

    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel, seed = seed)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat



