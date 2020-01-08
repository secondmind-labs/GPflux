# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numpy as np
import tensorflow as tf

from gpflux2.initializers import Initializer


class FeedForwardInitializer(Initializer):
    """
    Base object that initialises variational parameters to zero-mean, pointlike
    distributions, and defers the initialistion of the until data is available. When
    data is avaiable, it initialises the variables to be a random selection of the
    inputs
    """

    def __init__(self, q_sqrt_variance: float = 1e-2):
        super().__init__(init_at_predict=True)
        self.q_sqrt_variance = q_sqrt_variance

    def init_variational_params(self, q_mu, q_sqrt) -> None:
        """Initialise the variational parameter to a zero mean and """
        num_inducing_vars, num_output_dims = q_mu.shape
        q_mu_value = np.zeros((num_inducing_vars, num_output_dims))
        q_sqrt_value = (
            np.tile(np.eye(num_inducing_vars), (num_output_dims, 1, 1))
            * self.q_sqrt_variance
        )

        q_mu.assign(q_mu_value)
        q_sqrt.assign(q_sqrt_value)

    def init_inducing_variable(self, inducing_variable, input_data) -> None:
        data_rows = input_data.numpy().reshape(-1, input_data.shape[-1])  # [B, D]

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
