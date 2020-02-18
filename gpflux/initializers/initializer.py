# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    """Base object that initialises variational parameters and inducing points"""

    def __init__(self, init_at_predict: bool):
        self.init_at_predict = init_at_predict

    @abstractmethod
    def init_variational_params(self, q_mu, q_sqrt) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_inducing_variable(self, inducing_variable, inputs=None) -> None:
        raise NotImplementedError


class VariationalInitializer(Initializer):
    """
    Base object that initialises variational parameters to zero-mean, diagonal
    covariance pointlike distributions.
    """

    def __init__(self, init_at_predict: bool, q_sqrt_diagonal: float):
        super().__init__(init_at_predict=init_at_predict)
        self.q_sqrt_diagonal = q_sqrt_diagonal

    def init_variational_params(self, q_mu, q_sqrt) -> None:
        """
        Initialize the variational parameter to a zero mean and diagonal covariance
        """
        num_inducing_vars, num_output_dims = q_mu.shape
        q_mu_value = np.zeros((num_inducing_vars, num_output_dims))
        q_sqrt_value = (
            np.tile(np.eye(num_inducing_vars), (num_output_dims, 1, 1))
            * self.q_sqrt_diagonal
        )

        q_mu.assign(q_mu_value)
        q_sqrt.assign(q_sqrt_value)
