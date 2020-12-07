# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData

from gpflux.layers.likelihood_layer import LikelihoodLayer


class BayesianModel(tf.keras.Model):
    def __init__(
        self,
        X_input: TensorType,
        Y_input: TensorType,
        F_output: TensorType,
        likelihood_layer: LikelihoodLayer,
        num_data: int,
    ):
        y_output = likelihood_layer(F_output, targets=Y_input)
        super().__init__(inputs=[X_input, Y_input], outputs=[F_output, y_output])
        self.likelihood_layer = likelihood_layer
        self.X_input = X_input
        self.Y_input = Y_input
        self.F_output = F_output

        self.num_data = num_data

    def elbo(self, data: RegressionData) -> TensorType:
        """This is just a wrapper for explanatory value. Calling the model with training=True
        returns the predictive means and variances, but calculates the ELBO in the losses
        of the model"""

        X, Y = data
        # TF quirk: It looks like "__call__" does extra work before calling "call". We want this.
        _ = self.__call__((X, Y), training=True)
        return -tf.reduce_sum(self.losses * self.num_data)

    def predict(self, X: InputData) -> Tuple[MeanAndVariance, MeanAndVariance]:
        """Predict the output given just X values

        This method wraps around the standard call to a Keras model, but sadly mocks in the value of
        Y. This is clearly not ideal, and preferably we would execute only some parts of the graph.
        An alternative in future would just be to use tf.functions of subgraph Keras models,
        to perform prediction performantly."""
        return self.__call__([X, tf.cast(np.nan, default_float())], training=False)

    def predict_f(self, X: InputData) -> MeanAndVariance:
        return self.predict(X)[0]

    def predict_y(self, X: InputData) -> MeanAndVariance:
        return self.predict(X)[1]

    def predict_f_samples(
        self, X: InputData, num_samples: int, full_cov: bool = False, full_output_cov: bool = False,
    ) -> TensorType:
        if full_cov or full_output_cov:
            raise NotImplementedError
        mu, var = self.predict_f(X)
        mvn = tfp.distributions.MultivariateNormalDiag(mu, var)
        return mvn.sample(num_samples)
