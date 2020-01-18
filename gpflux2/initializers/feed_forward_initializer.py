# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import tensorflow as tf

from .initializers import VariationalInitializer


class FeedForwardInitializer(VariationalInitializer):
    """
    This initializer defers the initialization of the inducing variable until
    data is available. When data is avaiable, it initializes the variables to be
    a random selection of the inputs.
    """

    def __init__(self, q_sqrt_diagonal: float = 1e-2):
        super().__init__(init_at_predict=True, q_sqrt_diagonal=q_sqrt_diagonal)

    def init_inducing_variable(self, inducing_variable, inputs) -> None:
        data_rows = inputs.numpy().reshape(-1, inputs.shape[-1])  # [B, D]

        # HACK to deal with multioutput inducing variables
        if hasattr(inducing_variable, "inducing_variable_list"):
            inducing_variable_list = inducing_variable.inducing_variable_list
        elif hasattr(inducing_variable, "inducing_variable_shared"):
            inducing_variable_list = [inducing_variable.inducing_variable_shared]
        else:
            raise Exception

        for inducing_var in inducing_variable_list:
            choices = np.random.choice(np.arange(len(data_rows)), len(inducing_var),
                                       replace=False)
            initialization_data = data_rows[..., choices, :]
            inducing_var.Z.assign(initialization_data)
