# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np

import gpflow
from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian

from gpflux.layers import GPLayer
from gpflux.models import DeepGP


def _create_q_sqrt(M, L):
    return np.array([np.tril(np.random.randn(M, M)) for _ in range(L)])


class Data:
    D_in = 4
    D_out = 2
    N, M, Ns = 200, 50, 2
    X = np.random.randn(N, D_in)
    Z = np.random.randn(M, D_in)
    Y = np.random.randn(N, D_out)
    Xs = np.random.rand(Ns, D_in)

    Q_SQRT = _create_q_sqrt(M, D_out)
    Q_MU = np.random.randn(M, D_out)
    LSCs = np.random.randn(D_in) ** 2


def test_svgp_comparison(session_tf):
    # set up SVGP
    kern = RBF(Data.D_in, lengthscales=Data.LSCs, ARD=True)
    flow_model = SVGP(Data.X, Data.Y, kern, Gaussian(), Z=Data.Z)
    flow_model.q_mu = Data.Q_MU
    flow_model.q_sqrt = Data.Q_SQRT

    # set up GPflux
    feat = gpflow.features.InducingPoints(Data.Z)
    kern = RBF(Data.D_in, lengthscales=Data.LSCs, ARD=True)
    layer = GPLayer(kern, feat, Data.D_out)
    layer.q_mu = Data.Q_MU
    layer.q_sqrt = Data.Q_SQRT
    flux_model = DeepGP(Data.X, Data.Y, [layer])

    # test elbo
    np.testing.assert_almost_equal(flux_model.compute_log_likelihood(),
                                   flow_model.compute_log_likelihood())
    # test prediction
    m1, v1 = flow_model.predict_y(Data.Xs)
    m2, v2 = flux_model.predict_y(Data.Xs)

    np.testing.assert_array_almost_equal(m1, m2)
    np.testing.assert_array_almost_equal(v1, v2)
