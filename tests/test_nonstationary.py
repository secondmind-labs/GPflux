# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pytest
import numpy as np
from numpy.testing import assert_allclose

import gpflow

from gpflux.models.deep_gp import DeepGP
from gpflux.nonstationary import NonstationaryKernel
import gpflux


def test_positive_definite(session_tf):
    D = 2
    kern = NonstationaryKernel(gpflow.kernels.RBF(D), D)

    XL = np.random.rand(3, 2 * D)
    K1 = kern.compute_K_symm(XL)
    K2 = kern.compute_K(XL, XL)

    assert_allclose(K1, K2)

    np.linalg.cholesky(K1)


@pytest.mark.parametrize("lengthscales_dim", [1, 2])
def test_shapes(session_tf, lengthscales_dim):
    D = 2
    N = 5
    M = 6
    d = lengthscales_dim
    X1 = np.random.randn(N, D)
    X2 = np.random.randn(M, D)
    L1 = np.random.randn(N, d)
    L2 = np.random.randn(M, d)
    XL1 = np.hstack([X1, L1])
    XL2 = np.hstack([X2, L2])
    basekern = gpflow.kernels.RBF(D)
    kern = NonstationaryKernel(basekern, d)
    baseK = basekern.compute_K(X1, X2)
    K = kern.compute_K(XL1, XL2)
    assert baseK.shape == K.shape


def test_1D_equivalence(session_tf):
    D = 2
    kern = NonstationaryKernel(gpflow.kernels.RBF(D), D)
    kern_1D = NonstationaryKernel(gpflow.kernels.RBF(D), 1)

    XL1D = np.random.randn(3, D + 1)
    XL = np.concatenate([XL1D, np.tile(XL1D[:, -1, None], [1, D - 1])], 1)

    K = kern.compute_K_symm(XL)
    K_1D = kern_1D.compute_K_symm(XL1D)

    assert_allclose(K, K_1D)


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
