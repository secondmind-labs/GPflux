# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Keras Layer that wraps a likelihood, while containing the necessary operations
for training"""
from typing import Tuple, Union
from warnings import warn

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.likelihoods import Likelihood

from gpflux.layers.trackable_layer import TrackableLayer


class LikelihoodLayer(TrackableLayer):
    """
    A layer which wraps a GPflow likelihood, while providing a clear interface to
    help training.
    """

    def __init__(self, likelihood: Likelihood, returns_samples: bool = False):
        super().__init__(dtype=default_float())
        self.likelihood = likelihood
        self.returns_samples = returns_samples

    def call(
        self, inputs: TensorType, training: bool = False
    ) -> Union[Tuple[TensorType, TensorType], "LikelihoodOutputs"]:
        """
        This function will either return F and y samples or the parameters of the likelihood
        distribution.
        """
        if self.returns_samples:
            warn("Not implemented - returning the F samples.")
            f_samples = tf.convert_to_tensor(inputs)
            return f_samples, tf.convert_to_tensor(np.nan, f_samples.dtype)
        else:
            assert isinstance(inputs, tfp.distributions.MultivariateNormalDiag)

            F_mu = inputs.loc
            F_var = inputs.scale.diag

            Y_mean, Y_var = self.likelihood.predict_mean_and_var(F_mu, F_var)
            return LikelihoodOutputs(F_mu, F_var, Y_mean, Y_var)


class LikelihoodLoss(tf.keras.losses.Loss):
    """
    When training a Keras model using this loss function, the value of this loss is not logged
    explicitly (the layer-specific losses are logged, as is the overall model loss). To output
    this loss value explicitly, wrap this class in a `Metric` and add it to the model metrics.
    """

    def __init__(self, likelihood: Likelihood):
        super().__init__()

        self._likelihood = likelihood

    def call(
        self,
        y_true: TensorType,
        prediction: Union[Tuple[TensorType, TensorType], "LikelihoodOutputs"],
    ) -> TensorType:
        if isinstance(prediction, LikelihoodOutputs):
            F_mu, F_var = prediction.f_mu, prediction.f_var
            return -self._likelihood.variational_expectations(F_mu, F_var, y_true)
        else:
            # Note that y_pred is actually the f-samples. When sampling through the likelihood
            # is implemented this should be changed (because y_pred will then be y-samples).
            f_samples, _ = prediction
            return -self._likelihood.log_prob(f_samples, y_true)


class LikelihoodOutputs(tf.Module, metaclass=TensorMetaClass):
    """
    Class to encapsulate the outputs of the likelihood layer.

    This class contains the mean and variance of the marginal distribution of the final latent
    GP layer, as well as the mean and variance of the likelihood.

    This class includes the `TensorMetaClass` to make objects behave as a tf.Tensor. This is
    necessary so it can be returned from the `DistributionLambda` Keras layer.
    """

    def __init__(self, f_mu: TensorType, f_var: TensorType, y_mean: TensorType, y_var: TensorType):
        super(LikelihoodOutputs, self).__init__(name="distribution_params")

        self.f_mu = f_mu
        self.f_var = f_var
        self.y_mean = y_mean
        self.y_var = y_var

    def _value(
        self, dtype: tf.dtypes.DType = None, name: str = None, as_ref: bool = False
    ) -> TensorType:
        return self.y_mean

    @property
    def shape(self) -> TensorType:
        return self.y_mean.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.y_mean.dtype
