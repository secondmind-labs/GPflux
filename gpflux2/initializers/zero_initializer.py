# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Optional
import numpy as np
import tensorflow as tf

from .initializers import VariationalInitializer


class ZeroOneInitializer(VariationalInitializer):
    """
    Base object that initialises variational parameters to zero mean, identity
    covariance, and inducing points to a given value. This is useful in
    testing, as starting KL divergences are exactly 0.
    """

    def __init__(self, Z: Optional[np.ndarray] = None,
                 q_sqrt_diagonal: float = 1e-2):
        super().__init__(init_at_predict=False, q_sqrt_diagonal=q_sqrt_diagonal)
        self.Z = Z

    def init_inducing_variable(self, inducing_variable, inputs=None) -> None:
        if hasattr(inducing_variable, "inducing_variable_list"):
            inducing_variable_list = inducing_variable.inducing_variable_list
        elif hasattr(inducing_variable, "inducing_variable_shared"):
            inducing_variable_list = [inducing_variable.inducing_variable_shared]
        else:
            raise Exception

        for inducing_var in inducing_variable_list:
            Z = self.Z if self.Z is not None else tf.zeros_like(inducing_var.Z)
            inducing_var.Z.assign(Z)
