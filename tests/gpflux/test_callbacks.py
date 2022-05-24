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
from collections import defaultdict
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

import gpflow

import gpflux
from gpflux.experiment_support.tensorboard import tensorboard_event_iterator
from gpflux.helpers import construct_gp_layer


class CONFIG:
    hidden_dim = 11
    num_inducing = 13
    num_data = 7
    num_epochs = 3

    # model setting:
    likelihood_variance = 0.05


@pytest.fixture
def data() -> Tuple[np.ndarray, np.ndarray]:
    """Step function: f(x) = -1 for x <= 0 and 1 for x > 0."""
    X = np.linspace(-1, 1, CONFIG.num_data)
    Y = np.where(X > 0, np.ones_like(X), -np.ones_like(X))
    return (X.reshape(-1, 1), Y.reshape(-1, 1))


@pytest.fixture
def model_and_loss(data) -> Tuple[tf.keras.models.Model, tf.keras.losses.Loss]:
    """
    Builds a two-layer deep GP model.
    """
    X, Y = data
    num_data, input_dim = X.shape

    layer1 = construct_gp_layer(
        num_data, CONFIG.num_inducing, input_dim, CONFIG.hidden_dim, name="gp0"
    )

    output_dim = Y.shape[-1]
    layer2 = construct_gp_layer(
        num_data, CONFIG.num_inducing, CONFIG.hidden_dim, output_dim, name="gp1"
    )

    likelihood = gpflow.likelihoods.Gaussian(CONFIG.likelihood_variance)
    gpflow.set_trainable(likelihood.variance, False)

    X = tf.keras.Input((input_dim,))
    f1 = layer1(X)
    f2 = layer2(f1)

    # We add a dummy layer so that the likelihood variance is discovered as trainable:
    likelihood_container = gpflux.layers.TrackableLayer()
    likelihood_container.likelihood = likelihood
    y = likelihood_container(f2)

    loss = gpflux.losses.LikelihoodLoss(likelihood)
    return tf.keras.Model(inputs=X, outputs=y), loss


@pytest.mark.parametrize("update_freq", ["epoch", "batch"])
def test_tensorboard_callback(tmp_path, model_and_loss, data, update_freq):
    """Check the correct population of the TensorBoard event files"""

    tmp_path = str(tmp_path)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(CONFIG.num_data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model, loss = model_and_loss
    model.compile(optimizer=optimizer, loss=loss)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            patience=1,
            factor=0.95,
            verbose=1,
            min_lr=1e-6,
        ),
        # To write the LR to TensorBoard the `TensorBoard` callback needs to be
        # instantiated after the `ReduceLROnPlateau` callback.
        gpflux.callbacks.TensorBoard(tmp_path, update_freq=update_freq),
    ]
    history = model.fit(dataset, epochs=CONFIG.num_epochs, callbacks=callbacks)

    tb_files_pattern = f"{tmp_path}/train/events.out.tfevents*"  # notice the glob pattern

    # Maps tensorboard tags (e.g. kernel.variance) to list containing
    # their successive values during optimisation.
    records = defaultdict(list)  # Dict[str, list]

    # Loop over all events and add them to dict
    for event in tensorboard_event_iterator(tb_files_pattern):
        records[event.tag].append(event.value)

    # Keras adds a single event of `batch_2`, which we ignore.
    # It's not visible in the TensorBoard view, but it is in the event file.
    del records["batch_2"]

    expected_tags = {
        "epoch_lr",
        "epoch_loss",
        "epoch_gp0_prior_kl",
        "epoch_gp1_prior_kl",
        "self_tracked_trackables[1].kernel.kernel.lengthscales",
        "self_tracked_trackables[1].kernel.kernel.variance",
        "self_tracked_trackables[1]._self_tracked_trackables[1].kernel.lengthscales",
        "self_tracked_trackables[1]._self_tracked_trackables[1].kernel.variance",
        "self_tracked_trackables[2].kernel.kernel.lengthscales[0]",
        "self_tracked_trackables[2].kernel.kernel.lengthscales[1]",
        "self_tracked_trackables[2].kernel.kernel.lengthscales[2]",
        "self_tracked_trackables[2].kernel.kernel.variance",
        "self_tracked_trackables[2]._self_tracked_trackables[1].kernel.lengthscales[0]",
        "self_tracked_trackables[2]._self_tracked_trackables[1].kernel.lengthscales[1]",
        "self_tracked_trackables[2]._self_tracked_trackables[1].kernel.lengthscales[2]",
        "self_tracked_trackables[2]._self_tracked_trackables[1].kernel.variance",
        "self_tracked_trackables[3].likelihood.variance",
    }

    if Version(tf.__version__) < Version("2.8"):
        if update_freq == "batch":
            expected_tags |= {
                "batch_loss",
                "batch_gp0_prior_kl",
                "batch_gp1_prior_kl",
            }

    # Check all model variables, loss and lr are in tensorboard.
    assert set(records.keys()) == expected_tags

    # Check that length of each summary is correct.
    for record in records.values():
        assert len(record) == CONFIG.num_epochs

    # Check that recorded TensorBoard loss matches Keras history
    np.testing.assert_array_almost_equal(records["epoch_loss"], history.history["loss"], decimal=5)

    # Check correctness of fixed likelihood variance
    tag = ("layers[3].likelihood.variance",)
    assert all([v == CONFIG.likelihood_variance for v in records[tag]])
