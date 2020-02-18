# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import tensorflow as tf

from .initializer import VariationalInitializer


class FeedForwardInitializer(VariationalInitializer):
    """
    This initializer defers the initialization of the inducing variable until
    data is available. When data is avaiable, it initializes the variables to be
    a random selection of the inputs.
    """

    def __init__(self, q_sqrt_diagonal: float = 1e-2):
        super().__init__(init_at_predict=True, q_sqrt_diagonal=q_sqrt_diagonal)

    def init_inducing_variable(self, inducing_variable, inputs) -> None:
        data_rows = tf.reshape(inputs, (-1, inputs.shape[-1]))  # [N, D]

        # HACK to deal with multioutput inducing variables
        if hasattr(inducing_variable, "inducing_variable_list"):
            inducing_variable_list = inducing_variable.inducing_variable_list
        elif hasattr(inducing_variable, "inducing_variable_shared"):
            inducing_variable_list = [inducing_variable.inducing_variable_shared]
        else:
            raise Exception

        for inducing_var in inducing_variable_list:
            # TF quirk: tf.random.shuffle is not in-place
            initialization_data = tf.random.shuffle(inputs)[: len(inducing_var)]
            inducing_var.Z.assign(initialization_data)
