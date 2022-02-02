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

from typing import Optional

import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC

from gpflow.base import DType, TensorType

from gpflux.layers.basis_functions.fourier_features.random.base import RandomFourierFeatures


class QuasiRandomFourierFeatures(RandomFourierFeatures):

    r"""
    Quasi-random Fourier features (ORF) :cite:p:`yang2014quasi` for more
    efficient and accurate kernel approximations than random Fourier features.
    """

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        n_components, input_dim = shape  # M, D
        sampler = MultivariateNormalQMC(mean=np.zeros(input_dim))
        return sampler.random(n=n_components)  # shape [M, D]
