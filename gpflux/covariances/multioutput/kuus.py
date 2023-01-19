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


from gpflux.inducing_variables import (
    FallbackSharedIndependentDistributionalInducingVariables,
)
from gpflux.kernels import (
    DistributionalSharedIndependent
)

from gpflux.covariances.dispatch import Kuu


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
