# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""Module containing a Bayesian model."""
from typing import Union, List, Dict, Any

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import TensorType
from gpflow.models.training_mixins import InputData, RegressionData

from gpflux.layers import LikelihoodLoss, LikelihoodLayer
from gpflux.layers.likelihood_layer import LikelihoodOutputs


class BayesianModel(tf.keras.Model):
    def __init__(
        self,
        X_input: TensorType,
        F_output: TensorType,
        likelihood_layer: LikelihoodLayer,
        num_data: int,
    ):
        y_output = likelihood_layer(F_output)
        super().__init__(inputs=X_input, outputs=y_output)

        self.likelihood = likelihood_layer.likelihood
        self.num_data = num_data

        self._F_model = tf.keras.Model(inputs=X_input, outputs=F_output)

    def elbo(self, data: RegressionData) -> TensorType:
        """This is just a wrapper for explanatory value. Calling the model with training=True
        returns the predictive means and variances, but calculates the ELBO in the losses
        of the model"""

        X, Y = data
        # TF quirk: It looks like "__call__" does extra work before calling "call". We want this.
        outputs = self.__call__(X, training=True)
        model_loss = self.loss(Y, outputs)
        return -tf.reduce_sum((self.losses + model_loss) * self.num_data)

    def compile(
        self,
        optimizer: Union[tf.optimizers.Optimizer, str] = "rmsprop",  # Keras default
        loss: Union[tf.losses.Loss, str] = None,
        metrics: Union[tf.metrics.Metric, str] = None,
        loss_weights: TensorType = None,
        weighted_metrics: List[tf.metrics.Metric] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        if loss is None:
            loss = LikelihoodLoss(self.likelihood)
        super().compile(optimizer, loss, metrics, loss_weights, None, weighted_metrics, **kwargs)

    def predict_f(self, X: InputData) -> Union[tfp.distributions.Distribution, TensorType]:
        return self._F_model(X)

    def predict_y(self, X: InputData) -> Union[LikelihoodOutputs, TensorType]:
        return self.__call__(X)

    def predict_f_samples(
        self, X: InputData, num_samples: int, full_cov: bool = False, full_output_cov: bool = False,
    ) -> TensorType:
        if full_cov or full_output_cov:
            raise NotImplementedError
        distribution = self.predict_f(X)
        return distribution.sample(num_samples)
