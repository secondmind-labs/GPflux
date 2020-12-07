# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from typing import Optional

import tensorflow as tf

from gpflow import default_float
from gpflow.base import TensorType
from gpflow.inducing_variables import InducingPoints

from .initializer import Initializer


class FeedForwardInitializer(Initializer):
    """
    This initializer defers the initialization of the inducing variable until
    data is available. When data is available, it initializes the variables to
    be a random subset of the inputs. If not enough data is available, it will
    pad with draws from a unit-normal distribution.
    """

    def __init__(self) -> None:
        super().__init__(init_at_predict=True)

    def init_single_inducing_variable(
        self, inducing_variable: InducingPoints, inputs: Optional[TensorType] = None
    ) -> None:
        if inputs is None:
            raise ValueError(
                "FeedForwardInitializer requires `inputs` to be passed"
            )  # pragma: no cover

        num_inducing = inducing_variable.num_inducing
        num_data = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]

        # TF quirk: tf.random.shuffle is stateless (rather than in-place)
        data_rows = tf.random.shuffle(inputs)[:num_inducing]

        # When the number of inducing points is larger than the minibatch size,
        # pad with random values:
        # TODO: should we throw an error instead?
        num_extra_rows = tf.maximum(0, num_inducing - num_data)
        extra_rows = tf.random.normal((num_extra_rows, input_dim), dtype=default_float())
        initialization_data = tf.concat([data_rows, extra_rows], axis=0)

        inducing_variable.Z.assign(initialization_data)
