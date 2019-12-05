# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import numpy as np
import tensorflow as tf

from gpflux2.initializers import Initializer


class ZeroInitializer(Initializer):
    """
    Base object that initialises variational parameters to zero-mean, and inducing
    variables to 0. This is useful in testing, as starting KL divergences are exactly 0.
    """

    def __init__(self):
        super().__init__()
        self.deferred_init = False

    def init_variational_params(self, q_mu, q_sqrt) -> None:
        """Initialise the variational parameter to a zero mean and """
        num_inducing_vars, num_output_dims = q_mu.shape
        q_mu_value = np.zeros((num_inducing_vars, num_output_dims))
        q_sqrt_value = (
            np.tile(np.eye(num_inducing_vars), (num_output_dims, 1, 1))
        )

        q_mu.assign(q_mu_value)
        q_sqrt.assign(q_sqrt_value)

    def init_inducing_variable(self, inducing_variable, input_data=None) -> None:
        if hasattr(inducing_variable, "inducing_variable_list"):
            inducing_variable_list = inducing_variable.inducing_variable_list
        elif hasattr(inducing_variable, "inducing_variable_shared"):
            inducing_variable_list = [inducing_variable.inducing_variable_shared]
        else:
            raise Exception

        for inducing_var in inducing_variable_list:
            inducing_var.Z.assign(tf.zeros_like(inducing_var.Z))
