# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import pytest
import tensorflow as tf

import gpflow

from gpflux.invariance.kernels import Invariant, StochasticInvariant
from gpflux.invariance.orbits import QuantRotation, Rot90


@pytest.mark.parametrize("orbit,samples,orbit_kwargs, full_orbit, full_orbit_kwargs", [
    (Rot90, 7000, {"orbit_batch_size": 2}, None, {}),
    (QuantRotation, 1, {"orbit_batch_size": 8}, None, {}),  # Test the full batch size
    # (QuantRotation, 20000, {"orbit_batch_size": 5}, None, {}),
    # (Rotation, 3500, {"orbit_batch_size": 30, "use_stn": False}, QuantRotation,
    #  {"rotation_quantisation": 0.0625}),
    # (Rotation, 2000, {"orbit_batch_size": 30, "use_stn": True},
    # QuantRotation, {"rotation_quantisation": 0.125})
])
def test_stochastic_kernel_convergence(session_tf, orbit, full_orbit, samples, orbit_kwargs,
                                       full_orbit_kwargs):
    """
    Test whether the stochastic invariant kernel does actually return the
    correct unbiased estimates for both K and Kdiag.
    :param k:
    :param sk:
    """
    np.random.seed(234525)
    tf.set_random_seed(5673546)
    X = np.random.randn(3, 2 ** 2)
    k = Invariant(X.shape[1], gpflow.kernels.SquaredExponential(X.shape[1]),
                  orbit(**orbit_kwargs) if full_orbit is None else full_orbit(**full_orbit_kwargs))
    sk = StochasticInvariant(X.shape[1], gpflow.kernels.SquaredExponential(X.shape[1]),
                             orbit(**orbit_kwargs))

    K = k.compute_K_symm(X, X)
    sK = sum([sk.compute_K_symm(X, X) for _ in range(samples)]) / samples
    pd = np.max(np.abs(K - sK) / K * 100.0)
    assert pd < 0.5

    dK = sum([sk.compute_Kdiag(X) for _ in range(samples)]) / samples
    pd = np.max(np.abs(k.compute_Kdiag(X) - dK) / dK * 100.0)
    assert pd < 0.5
