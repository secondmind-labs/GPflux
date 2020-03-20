# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Implements the Doubly Stochastic Deep Gaussian Process by Salimbeni & Deisenroth (2017)
http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes
"""

import argparse
from typing import Tuple, List, Sequence, Optional

import numpy as np
import tensorflow as tf
import tqdm

from gpflux.architectures.dgp_2017 import build_deep_gp_2017, Config as DGP2017Config
from gpflux.helpers import make_dataclass_from_class

from automl.model_training.custom_model import CustomModel, register_custom_model
from automl.reports import ModelDescriptionTable
from automl_implementations.benchmarks.research_env.gpflux.gpflux_support import (
    art_aware_main,
    get_experiment_arguments,
    local_main,
)
from utilities import pio_logging

LOG = pio_logging.logger(__name__)
tf.keras.backend.set_floatx("float64")


class Config:
    """
    Configuration class
    """

    num_inner_layers = 2  # 1, 2, 3, 4 ... TODO: search over this!

    num_inducing: int = 100
    likelihood_noise_variance: float = 0.01  # from paper; code says 0.05
    between_layer_noise_variance: float = 1e-5  # from paper; code says 2e-6
    inner_layer_qsqrt_factor: float = 1e-5  # from code; paper says sqrt(1e-5)
    white: bool = True  # code/paper says False

    maxiter: int = 20000  # for small-medium; large: 100,000; taxi: 500,000
    adam_learning_rate: float = 0.01
    minibatch_size: Optional[int] = 10000


class DoublyStochasticDGP2017(CustomModel):
    """
    Implements the Doubly Stochastic Deep Gaussian Process by Salimbeni & Deisenroth (2017)
    http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes
    """

    @property
    def number_of_parameters(self) -> int:
        """
        Compute number of parameters in the model. Used to approximate model
        complexity in BIC.  Variational parameters should not be included here
        as the number of parameters here refers to model hyperparameters, NOT
        variational parameters. The former reflect additional model complexity
        and should be penalised in model selection. The latter capture
        approximation error - the more we have the less the approximation error
        with a corresponding increase in computational complexity.
        """
        return -1

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Construct and train a GPflux model.
        """
        # We have to help PyLint out - it doesn't understand it's working with GPflow2 here
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

        # NOTE: if you're using normalisation, the kernel parameters live in the normalised space

        # TODO: print("Configuration")
        # pprint(vars(Config))

        ### build
        model_config = make_dataclass_from_class(DGP2017Config, Config)

        _, input_dim = X.shape
        L = min(30, input_dim)
        layer_dims = [input_dim] + [L] * Config.num_inner_layers + [1]

        self.model = build_deep_gp_2017(X, layer_dims, model_config)
        # TODO: print_summary(self.model)

        ### train
        if self.options.get("optimize", True):
            self._optimize(X, Y)

        LOG.info("Optimisation complete")

    def _optimize(self, X, Y, callbacks=None):
        num_data, _ = X.shape
        adam_opt = tf.optimizers.Adam(learning_rate=Config.adam_learning_rate)

        data = (X, Y)
        dataset_tuple = (data, Y)  # (inputs, targets); we need inputs=(X, Y) to pass to latent layers, and targets=Y for keras

        train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).prefetch(num_data).repeat()
        if Config.minibatch_size is not None:
            batch_size = min(Config.minibatch_size, num_data)
            train_dataset = (train_dataset
                .shuffle(num_data)
                .batch(batch_size)
            )

        num_epochs = Config.maxiter  # TODO fix

        # TODO: print("Before optimization:", self.model.elbo(data))

        self.model.compile(optimizer=adam_opt)
        _ = self.model.fit(train_dataset, steps_per_epoch=1, epochs=num_epochs, callbacks=callbacks)

        # TODO: print("After optimization:", self.model.elbo(data))

    def predict_mean_and_var(self, Xnew: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict_y(Xnew)

    def get_model_description(self) -> ModelDescriptionTable:
        return ModelDescriptionTable(name="DGP2017", descriptions=[])


register_custom_model("DGP2017", DoublyStochasticDGP2017)


if __name__ == "__main__":
    args = get_experiment_arguments()
    args.configs = ["DGP2017"]
    art_aware_main(args)
