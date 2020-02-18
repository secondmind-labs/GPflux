# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pytest
import numpy as np
from numpy.testing import assert_allclose

import gpflow
from gpflux.nonstationary import NonstationaryKernel


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
