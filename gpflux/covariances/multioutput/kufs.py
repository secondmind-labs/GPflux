#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tensorflow as tf
import tensorflow_probability as tfp



from gpflux.inducing_variables import SharedIndependentDistributionalInducingVariables


from gpflow.inducing_variables import (
    SharedIndependentInducingVariables,
)


from gpflux.kernels import DistributionalSharedIndependent 
from gpflow.covariances.dispatch import Kuf

@Kuf.register(SharedIndependentDistributionalInducingVariables, DistributionalSharedIndependent, tfp.distributions.MultivariateNormalDiag)
def Kuf_shared_shared_distributional(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: DistributionalSharedIndependent,
    Xnew: tfp.distributions.MultivariateNormalDiag,
    *,
    seed:int = None
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew, seed = seed)  # [M, N]
