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


import numpy as np

import gpflow

import gpflux
from gpflux.models.deep_gp import DeepGP
from gpflux.nonstationary import NonstationaryKernel


def test_nonstationary_gp_1d(session_tf):
    """
    This test build a deep nonstationary GP model consisting of 3 layers,
    and checks if the model can be optimized.
    """
    D_in = D1 = D2 = D_out = 1
    X = np.linspace(0, 10, 500).reshape(-1, 1)
    Y = np.random.randn(500, 1)
    Xlam = np.c_[X, np.zeros_like(X)]

    # Layer 1
    Z1 = X.copy()
    feat1 = gpflow.features.InducingPoints(Z1)
    kern1 = gpflow.kernels.RBF(D_in, lengthscales=0.1)
    layer1 = gpflux.layers.GPLayer(kern1, feat1, D1)

    # Layer 2
    Z2 = Xlam.copy()
    feat2 = gpflow.features.InducingPoints(Z2)
    kern2 = NonstationaryKernel(gpflow.kernels.RBF(D1), D_in, scaling_offset=-0.1,
                                positivity=gpflow.transforms.positive.forward_tensor)
    layer2 = gpflux.layers.NonstationaryGPLayer(kern2, feat2, D2)

    # Layer 3
    Z3 = Xlam.copy()
    feat3 = gpflow.features.InducingPoints(Z3)
    kern3 = NonstationaryKernel(gpflow.kernels.RBF(D2), D1, scaling_offset=-0.1,
                                positivity=gpflow.transforms.positive.forward_tensor)
    layer3 = gpflux.layers.NonstationaryGPLayer(kern3, feat3, D_out)

    model = DeepGP(X, Y, [layer1, layer2, layer3])

    # minimize
    likelihood_before_opt = model.compute_log_likelihood()
    gpflow.train.AdamOptimizer(0.01).minimize(model, maxiter=100)
    likelihood_after_opt = model.compute_log_likelihood()

    assert likelihood_before_opt < likelihood_after_opt
