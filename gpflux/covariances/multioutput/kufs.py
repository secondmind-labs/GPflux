import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union


from gpflow.base import TensorLike, TensorType

from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import (
    MultioutputKernel,
    SharedIndependent,
)

from gpflux.covariances.dispatch import Kuf


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def Kuf_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]
