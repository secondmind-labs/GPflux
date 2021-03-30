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
import pytest
import tensorflow as tf

from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.exceptions import EncoderInitializationError

num_data = 200
latent_dim = 3


def test_shapes():
    seed = 300
    np.random.seed(seed)
    dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim)

    # Tests shapes
    assert np.all(tf.shape(dp_encoder.means) == (num_data, latent_dim))
    assert np.all(tf.shape(dp_encoder.stds) == (num_data, latent_dim))

    # Tests values
    np.random.seed(seed)
    expected_means = 0.01 * np.random.randn(num_data, latent_dim)
    expected_stds = 1e-5 * np.ones_like(expected_means)
    np.testing.assert_equal(dp_encoder.means.numpy(), expected_means)
    np.testing.assert_allclose(dp_encoder.stds.numpy(), expected_stds, rtol=1e-11)


@pytest.mark.parametrize("means", [None, np.random.randn(num_data, latent_dim)])
def test_call(means):
    dp_encoder = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)
    encoder_means, encoder_std = dp_encoder(inputs=None)

    assert encoder_means is dp_encoder.means
    assert encoder_std is dp_encoder.stds

    assert np.all(tf.shape(encoder_means) == (num_data, latent_dim))
    if means is not None:
        np.testing.assert_array_equal(encoder_means.numpy(), means)


def test_bad_shapes():
    means = np.random.randn(num_data, latent_dim + 4)
    with pytest.raises(EncoderInitializationError):
        _ = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)

    means = np.random.randn(num_data + 1, latent_dim)
    with pytest.raises(EncoderInitializationError):
        _ = DirectlyParameterizedNormalDiag(num_data, latent_dim, means)
