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
r"""
This module provides the base implementation for vanilla Deep GP models.
By "Vanilla" we refer to a model that can only contain standard :class:`GPLayer`\s,
this model does not support keras layers, latent variable layers, etc.
"""

from typing import List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.base import TensorType

from gpflux.experiment_support.plotting import plot_layers
from gpflux.layers import GPLayer, LikelihoodLayer, TrackableLayer
from gpflux.models.deep_gp import DeepGP
from gpflux.sampling.sample import Sample


class VanillaDeepGP(DeepGP):
    def __init__(
        self,
        gp_layers: List[GPLayer],
        likelihood: Union[
            LikelihoodLayer, gpflow.likelihoods.Likelihood
        ],  # fully-qualified for autoapi
        *,
        input_dim: Optional[Union[int, Sequence[int]]] = None,
        target_dim: Optional[int] = None,
        default_model_class: Type[tf.keras.Model] = tf.keras.Model,
        num_data: Optional[int] = None,
    ):
        """
        :param gp_layers: The layers ``[f₁, f₂, …, fₙ]`` describing the latent
            function ``f(x) = fₙ(⋯ (f₂(f₁(x))))``.
        :param likelihood: The layer for the likelihood ``p(y|f)``. If this is a
            GPflow likelihood, it will be wrapped in a :class:`~gpflux.layers.LikelihoodLayer`.
            Alternatively, you can provide a :class:`~gpflux.layers.LikelihoodLayer` explicitly.
        :param input_dim: The input dimensionality.
        :param target_dim: The target dimensionality.
        :param default_model_class: The default for the *model_class* argument of
            :meth:`as_training_model` and :meth:`as_prediction_model`;
            see the :attr:`default_model_class` attribute.
        :param num_data: The number of points in the training dataset; see the
            :attr:`num_data` attribute.
            If you do not specify a value for this parameter explicitly, it is automatically
            detected from the :attr:`~gpflux.layers.GPLayer.num_data` attribute in the GP layers.
        """
        # if not all([isinstance(layer, GPLayer) for layer in gp_layers]):
        #     raise ValueError(
        #         "`VanillaDeepGP` can only be build out of `GPLayer`s. "
        #         "Use `DeepGP` for a hybrid model with, for example, keras layers, "
        #         "latent variable layers and GP layers."
        #     )

        super().__init__(
            gp_layers,
            likelihood,
            input_dim=input_dim,
            target_dim=target_dim,
            default_model_class=default_model_class,
            num_data=num_data,
        )
    
    def as_dnn_model(self):
        """
        Creates a Neural Network equivalent of the Deep GP model
        """
        outputs = self.inputs
        for layer in self.f_layers:
            original_convert_to_tensor = layer._convert_to_tensor_fn
            layer._convert_to_tensor_fn = tfp.distributions.Distribution.mean,
            outputs = tf.convert_to_tensor(layer(outputs, training=False))
            layer._convert_to_tensor_fn = original_convert_to_tensor

        model = tf.keras.Model(inputs=self.inputs, outputs=outputs)
        return model

        # likelihood = self.likelihood_layer.likelihood
        # likelihood_container = TrackableLayer()
        # likelihood_container.likelihood = likelihood
        # y = likelihood
        # model = tf.keras.model.Model(=self.inputs, f)
        # return model
        # loss = gpflux.losses.LikelihoodLoss(likelihood)
        # model.compile(loss=loss, optimizer="adam")
        # outputs = self.call(self.inputs)

    def fit(
        self,
        X: TensorType,
        Y: TensorType,
        *,
        batch_size: Optional[int] = 32,
        learning_rate: float = 0.01,
        epochs: int = 128,
        verbose: int = 0,
    ) -> tf.keras.callbacks.History:
        """
        Compile and fit the model on training data (X, Y). Optimization
        uses Adam optimizer with decaying learning rate. This method wraps
        :meth:`tf.keras.Model.compile` and :meth:`tf.keras.Model.fit`.

        .. note:
            Parameter docs copied from Keras documention.

        :param X: Input data. A Numpy array (or array-like).
        :param Y: Target data. Like the input data `X`.
        :param batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            If set to `None`, will use size of dataset.
        :param epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `X` and `Y`
            data provided.
            If unspecified, `epochs` will default to 128.
        :param verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).

        :returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        training_model = self.as_training_model()
        training_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate))

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                "loss", factor=0.95, patience=3, min_lr=1e-6, verbose=verbose
            ),
        ]

        history = training_model.fit(
            {"inputs": X, "targets": Y},
            batch_size=batch_size or len(X),
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    def predict_f(self, X: TensorType) -> Tuple[TensorType, TensorType]:
        prediction_model = self.as_prediction_model(X)
        output = prediction_model.predict(X)
        return output.f_mean, output.f_var

    def predict_y(self, X: TensorType) -> Tuple[TensorType, TensorType]:
        prediction_model = self.as_prediction_model(X)
        output = prediction_model.predict(X)
        return output.y_mean, output.y_var

    def sample(self) -> Sample:
        function_draws = [layer.sample() for layer in self.f_layers]

        class ChainedSample(Sample):
            """ This class chains samples from consecutive layers. """

            def __call__(self, X: TensorType) -> tf.Tensor:
                self.inner_layers = []
                for f in function_draws:
                    X = f(X)
                    self.inner_layers.append(X)
                return X

        return ChainedSample()

    def plot(
        self, X: TensorType, Y: TensorType, num_test_points: int = 100, x_margin: float = 3.0
    ) -> Tuple[plt.Figure, np.ndarray]:

        if X.shape[1] != 1:
            raise NotImplementedError("DeepGP plotting is only supported for 1D models.")

        means, covs, samples = [], [], []

        X_test = np.linspace(X.min() - x_margin, X.max() + x_margin, num_test_points).reshape(-1, 1)
        layer_input = X_test

        for layer in self.f_layers:
            layer.full_cov = True
            layer.num_samples = 5
            layer_output = layer(layer_input)

            mean = layer_output.mean()
            cov = layer_output.covariance()
            sample = tf.convert_to_tensor(layer_output)  # generates num_samples samples...

            print(mean.shape)
            print(cov.shape)
            print(sample.shape)

            layer_input = sample[0]  # for the next layer

            means.append(mean.numpy().T)  # transpose to go from [1, N] to [N, 1]
            covs.append(cov.numpy())
            samples.append(sample.numpy())

        fig, axes = plot_layers(X_test, means, covs, samples)

        if axes.ndim == 1:
            axes[-1].plot(X, Y, "kx")
        elif axes.ndim == 2:
            axes[-1, -1].plot(X, Y, "kx")

        return fig, axes
