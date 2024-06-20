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
"""
This module provides the `LikelihoodLoss` adapter to use GPflow's
:class:`~gpflow.likelihoods.Likelihood` implementations as a
`tf.keras.losses.Loss`.
"""
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.base import TensorType
from gpflow.keras import tf_keras

from gpflux.types import unwrap_dist


class LikelihoodLoss(tf_keras.losses.Loss):
    r"""
    This class is a `tf.keras.losses.Loss` implementation that wraps a GPflow
    :class:`~gpflow.likelihoods.Likelihood` instance.

    When the prediction (last-layer output) is a
    :class:`~tfp.distributions.Distribution` ``q(f)``, calling this loss
    returns the negative variational expectation :math:`-\mathbb{E}_{q(f)}[\log
    p(y|f)]`. When the prediction is a `tf.Tensor`, calling this loss returns
    the negative log-probability :math:`-\log p(y|f)`.

    When you use this loss function in training a Keras model, the value of
    this loss is not logged explicitly (in contrast, the layer-specific losses
    are logged, as is the overall model loss).
    To output this loss value explicitly, wrap this class in a
    `tf.keras.metrics.Metric` and add it to the model metrics.

    .. note::

        Use **either** this `LikelihoodLoss` (e.g. together with a
        `tf.keras.Sequential` model) **or**
        :class:`~gpflux.layers.LikelihoodLayer` (together with
        `gpflux.models.DeepGP`). Do **not** use both at once because this would
        add the loss twice.
    """

    def __init__(self, likelihood: gpflow.likelihoods.Likelihood):
        """
        :param likelihood: the GPflow likelihood object to use.

            .. note:: If you want to train any parameters of the likelihood
                (e.g. likelihood variance), you must include the likelihood as
                an attribute on a :class:`~gpflux.layers.TrackableLayer`
                instance that is part of your model. (This is not required when
                instead you use a :class:`gpflux.layers.LikelihoodLayer`
                together with :class:`gpflux.models.DeepGP`.)
        """
        super().__init__()

        self.likelihood = likelihood

    def call(
        self,
        y_true: TensorType,
        f_prediction: Union[TensorType, tfp.distributions.MultivariateNormalDiag],
    ) -> tf.Tensor:
        """
        Note that we deviate from the Keras Loss interface by calling the
        second argument *f_prediction* rather than *y_pred*.
        """
        no_X = None
        if isinstance(unwrap_dist(f_prediction), tfp.distributions.MultivariateNormalDiag):

            F_mu = f_prediction.loc
            F_var = f_prediction.scale.diag ** 2
            return -self.likelihood.variational_expectations(no_X, F_mu, F_var, y_true)
        else:  # Tensor
            f_samples = f_prediction
            return -self.likelihood.log_prob(no_X, f_samples, y_true)
