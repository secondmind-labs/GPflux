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
A Keras Layer that wraps a likelihood, while containing the necessary operations
for training.
"""
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.likelihoods import Likelihood

from gpflux.layers.trackable_layer import TrackableLayer
from gpflux.types import unwrap_dist


class LikelihoodLayer(TrackableLayer):
    r"""
    A Keras layer that wraps a GPflow :class:`~gpflow.likelihoods.Likelihood`. This layer expects a
    `tfp.distributions.MultivariateNormalDiag` as its input, describing ``q(f)``.
    When training, calling this class computes the negative variational expectation
    :math:`-\mathbb{E}_{q(f)}[\log p(y|f)]` and adds it as a layer loss.
    When not training, it computes the mean and variance of ``y`` under ``q(f)``
    using :meth:`~gpflow.likelihoods.Likelihood.predict_mean_and_var`.

    .. note::

        Use **either** this `LikelihoodLayer` (together with
        `gpflux.models.DeepGP`) **or** `LikelihoodLoss` (e.g. together with a
        `tf.keras.Sequential` model). Do **not** use both at once because
        this would add the loss twice.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood

    def call(
        self,
        inputs: tfp.distributions.MultivariateNormalDiag,
        targets: Optional[TensorType] = None,
        training: bool = None,
    ) -> "LikelihoodOutputs":
        """
        When training (``training=True``), this method computes variational expectations
        (data-fit loss) and adds this information as a layer loss.
        When testing (the default), it computes the posterior mean and variance of ``y``.

        :param inputs: The output distribution of the previous layer. This is currently
            expected to be a :class:`~tfp.distributions.MultivariateNormalDiag`;
            that is, the preceding :class:`~gpflux.layers.GPLayer` should have
            ``full_cov=full_output_cov=False``.
        :returns: a `LikelihoodOutputs` tuple with the mean and variance of ``f`` and,
            if not training, the mean and variance of ``y``.

        .. todo:: Turn this layer into a
            :class:`~tfp.layers.DistributionLambda` as well and return the
            correct :class:`~tfp.distributions.Distribution` instead of a tuple
            containing mean and variance only.
        """
        # TODO: add support for non-distribution inputs? or other distributions?
        assert isinstance(unwrap_dist(inputs), tfp.distributions.MultivariateNormalDiag)
        no_X = None
        F_mean = inputs.loc
        F_var = inputs.scale.diag ** 2

        if training:
            assert targets is not None
            # TODO: re-use LikelihoodLoss to remove code duplication
            loss_per_datapoint = tf.reduce_mean(
                -self.likelihood.variational_expectations(no_X, F_mean, F_var, targets)
            )
            Y_mean = Y_var = None
        else:
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
            Y_mean, Y_var = self.likelihood.predict_mean_and_var(no_X, F_mean, F_var)

        self.add_loss(loss_per_datapoint)

        # TODO: turn this layer into a DistributionLambda as well and
        # return correct distribution, not a mean/variance tuple
        return LikelihoodOutputs(F_mean, F_var, Y_mean, Y_var)


class LikelihoodOutputs(tf.Module, metaclass=TensorMetaClass):
    """
    This class encapsulates the outputs of a :class:`~gpflux.layers.LikelihoodLayer`.

    It contains the mean and variance of the marginal distribution of the final latent
    :class:`~gpflux.layers.GPLayer`, as well as the mean and variance of the likelihood.

    This class includes the `TensorMetaClass
    <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/util/deferred_tensor.py#L81>`_
    to make objects behave as a `tf.Tensor`. This is necessary so that it can be
    returned from the `tfp.layers.DistributionLambda` Keras layer.
    """

    def __init__(
        self,
        f_mean: TensorType,
        f_var: TensorType,
        y_mean: Optional[TensorType],
        y_var: Optional[TensorType],
    ):
        super().__init__(name="likelihood_outputs")

        self.f_mean = f_mean
        self.f_var = f_var
        self.y_mean = y_mean
        self.y_var = y_var

    def _value(
        self, dtype: tf.dtypes.DType = None, name: str = None, as_ref: bool = False
    ) -> tf.Tensor:
        return self.f_mean

    @property
    def shape(self) -> tf.Tensor:
        return self.f_mean.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.f_mean.dtype
