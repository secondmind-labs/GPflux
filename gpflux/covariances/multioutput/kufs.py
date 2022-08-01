import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union


from gpflow.base import TensorLike, TensorType
from gpflux.inducing_variables import SharedIndependentDistributionalInducingVariables


from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    MultioutputKernel,
    SharedIndependent,
)

from gpflux.kernels import DistributionalSharedIndependent 
from gpflux.covariances.dispatch import Kuf


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def Kuf_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SharedIndependentDistributionalInducingVariables, DistributionalSharedIndependent, tfp.distributions.MultivariateNormalDiag)
def Kuf_shared_shared_distributional(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: DistributionalSharedIndependent,
    Xnew: tfp.distributions.MultivariateNormalDiag,
    *,
    seed:int = None
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew, seed = seed)  # [M, N]
