# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import pytest

import tensorflow as tf

import tensorflow.keras as keras

from gpflux2.encoders import DirectlyParameterizedNormalDiag
from gpflux2.exceptions import EncoderInitializationError


def test_shapes():
    seed = 50
    num_data, latent_dim = 200, 3
    np.random.seed(300)
    dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim)

    # Tests shapes
    assert np.all(tf.shape(dp_encoder.means) == (num_data, latent_dim))
    assert np.all(tf.shape(dp_encoder.std) == (num_data, latent_dim))

    # Tests values
    np.random.seed(300)
    assert np.all(dp_encoder.means == np.zeros((num_data, latent_dim)))
    np.testing.assert_allclose(
        dp_encoder.std.numpy(), 1e-5 * np.ones_like(dp_encoder.means)
    )


def test_call():
    num_data, latent_dim = 200, 3

    means = np.random.randn(num_data, latent_dim)
    dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)
    encoder_means, encoder_std = dp_encoder(inputs=None)

    assert np.all(tf.shape(encoder_means) == (num_data, latent_dim))
    assert np.all(means == encoder_means)
    assert encoder_means is dp_encoder.means
    assert encoder_std is dp_encoder.std


def test_bad_shapes():
    num_data, latent_dim = 200, 3

    means = np.random.randn(num_data, latent_dim + 4)
    with pytest.raises(EncoderInitializationError):
        dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)

    means = np.random.randn(num_data + 1, latent_dim)
    with pytest.raises(EncoderInitializationError):
        dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)