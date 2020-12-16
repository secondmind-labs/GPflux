# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from typing import List, Optional

import tensorflow as tf

from gpflow.base import TensorType

from gpflux.layers.gp_layer import GPLayer
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models.bayesian_model import BayesianModel
from gpflux.sampling.sample import Sample


class DeepGP(BayesianModel):
    def __init__(
        self,
        gp_layers: List[GPLayer],
        likelihood_layer: LikelihoodLayer,
        input_dim: Optional[int] = None,
    ):
        X_input = tf.keras.Input((input_dim,))

        F_output = X_input
        for gp_layer in gp_layers:
            assert isinstance(gp_layer, GPLayer)
            F_output = gp_layer(F_output)
        num_data = gp_layers[0].num_data

        super().__init__(X_input, F_output, likelihood_layer, num_data)

        self.gp_layers = gp_layers

    def sample(self) -> Sample:
        function_draws = [layer.sample() for layer in self.gp_layers]

        class ChainedSample(Sample):
            """Chains samples from consecutive layers."""

            def __call__(self, X: TensorType) -> TensorType:
                for f in function_draws:
                    X = f(X)
                return X

        return ChainedSample()
