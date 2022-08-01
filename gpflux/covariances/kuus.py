# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflow_probability as tfp

from typing import Any, Union, Optional

from gpflow.config import default_float
from gpflow.inducing_variables import InducingPoints

from gpflow.base import TensorLike
from gpflux.inducing_variables import DistributionalInducingPoints
from gpflow.kernels import Kernel
from gpflux.kernels import DistributionalKernel
from gpflux.covariances.dispatch import Kuu


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, *, jitter: float = 0.0
) -> tf.Tensor:
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(DistributionalInducingPoints, DistributionalKernel)
def Kuu_kernel_distributionalinducingpoints(
    inducing_variable: DistributionalInducingPoints, kernel: DistributionalKernel, *, jitter: float = 0.0, seed: int = None
) -> tf.Tensor:

    # Create instance of tfp.distributions.MultivariateNormalDiag so that it works with underpinning methods from kernel
    distributional_inducing_points = tfp.distributions.MultivariateNormalDiag(loc = inducing_variable.Z_mean,
        scale_diag = tf.sqrt(inducing_variable.Z_var))
    Kzz = kernel(distributional_inducing_points, seed = seed)    
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz