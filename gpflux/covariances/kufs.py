import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Optional, Union

from gpflow.base import TensorLike, TensorType
from gpflow.inducing_variables import InducingPoints
from gpflux.inducing_variables import DistributionalInducingPoints
from gpflow.kernels import Kernel
from gpflux.covariances.dispatch import Kuf
from gpflux.kernels import DistributionalKernel



@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew)

@Kuf.register(DistributionalInducingPoints, DistributionalKernel, tfp.distributions.MultivariateNormalDiag)
def Kuf_kernel_distributionalinducingpoints( 
    inducing_variable: DistributionalInducingPoints, 
    kernel: DistributionalKernel, 
    Xnew: TensorType,
    *,
    seed:int = None) -> tf.Tensor:

    # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel

    assert isinstance(Xnew, tfp.distributions.MultivariateNormalDiag)

    distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
        scale_diag = tf.sqrt(inducing_variable.Z_var))
    
    return kernel(distributional_inducing_points, Xnew, seed = seed)
    


