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


from typing import NamedTuple, Optional

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import Param, Parameterized, params_as_tensors, settings
from gpflow.conditionals import sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero

from gpflux.kernels import NonstationaryKernel
from gpflux.layers.gp_layer import GPLayer
from gpflux.types import TensorLike


class NonstationaryGPLayer(GPLayer):
    def __init__(self,
                 kernel: NonstationaryKernel,
                 feature: gpflow.features.InducingFeature,
                 *args, **kwargs):
        assert isinstance(kernel, NonstationaryKernel)
        if isinstance(feature, gpflow.features.InducingPoints):
            assert feature.Z.shape[1] == kernel.input_dim
        super().__init__(kernel, feature, *args, **kwargs)

    @params_as_tensors
    def propagate(self, H, *, X=None, **kwargs):
        """
        Concatenates original input X with output H of the previous layer
        for the non-stationary kernel: the latter will be interpreted as
        lengthscales.

        :param H: input to this layer [N, P]
        :param X: original input [N, D]
        """
        XH = tf.concat([X, H], axis=1)
        return super().propagate(XH, **kwargs)
