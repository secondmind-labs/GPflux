# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import pytest

import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras

from gpflow.kernels import RBF

from gpflux.layers import LatentVariableAugmentationLayer, LatentVariableLayer
from gpflux.encoders import DirectlyParameterizedNormalDiag

tf.keras.backend.set_floatx("float64")


@pytest.fixture
def test_data():
    x_dim, y_dim, w_dim = 2, 1, 2
    points = 200
    x_data = np.random.random((points, x_dim)) * 5
    w_data = np.random.random((points, w_dim))
    w_data[: (points // 2), :] = 0.2 * w_data[: (points // 2), :] + 5

    input_data = np.concatenate([x_data, w_data], axis=1)
    y_data = np.random.multivariate_normal(
        mean=np.zeros(points), cov=RBF(variance=0.1).K(input_data), size=y_dim
    ).T
    return x_data[:, :x_dim], y_data


@pytest.mark.parametrize("w_dim", [1, 5])
def test_augmentation(test_data, w_dim):
    x_data, y_data = test_data
    num_data, x_dim = x_data.shape

    means = np.random.randn(num_data, w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    # Prior is N(0, I)
    prior_mean = np.zeros(w_dim)
    prior_var = np.ones(w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)

    # Set Seed
    tf.random.set_seed(0)
    np.random.seed(0)

    lv_augmented = LatentVariableAugmentationLayer(encoder=encoder, prior=prior)
    sample_augmented = lv_augmented(test_data, seed=0)

    # Reset Seed
    tf.random.set_seed(0)
    np.random.seed(0)

    lv_simple = LatentVariableLayer(encoder=encoder, prior=prior)
    sample_simple = lv_simple(tf.concat(test_data, axis=-1), seed=0)

    assert np.all(sample_augmented[:, :x_dim] == x_data)
    assert np.all(sample_augmented[:, x_dim:] == sample_simple)


@pytest.mark.parametrize("w_dim", [1, 5])
def test_add_losses(w_dim):
    num_data = 400
    x_dim = 3
    y_dim = 1
    means = np.random.randn(num_data, w_dim)
    encoder = DirectlyParameterizedNormalDiag(num_data, w_dim, means)

    # Prior is N(0, I)
    prior_mean = np.zeros(w_dim)
    prior_var = np.ones(w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_mean, prior_var)

    lv = LatentVariableAugmentationLayer(encoder=encoder, prior=prior)

    inputs = (
        tf.zeros((num_data, x_dim), dtype=tf.float64),
        tf.zeros((num_data, y_dim), dtype=tf.float64),
    )

    _ = lv(inputs=inputs)
    assert lv.losses == [0.0]

    _ = lv(inputs=inputs, training=True)
    _, posteriors = lv.sample_posteriors(recognition_data=None)
    local_kls = [tf.reduce_mean(lv.local_kls(posteriors))]
    np.testing.assert_allclose(lv.losses, local_kls)
    assert lv.losses > [0.0]

