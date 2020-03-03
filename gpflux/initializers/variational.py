# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


SMALL_DIAGONAL_STDDEV = (
    1e-2  # small variance for inner layers gives a more sensible DGP initialization
)


class VariationalInitializer(ABC):
    """
    Base class for initializing the variational parameters.
    """

    @abstractmethod
    def init_variational_params(self, q_mu, q_sqrt) -> None:
        raise NotImplementedError


class MeanFieldVariationalInitializer(VariationalInitializer):
    """
    Initialises variational parameters to zero-mean, diagonal covariance
    pointlike distributions. By default initializes to a very small variance on
    the diagonal.
    """

    def __init__(self, q_sqrt_diagonal: Optional[float] = SMALL_DIAGONAL_STDDEV):
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


class ZeroOneVariationalInitializer(MeanFieldVariationalInitializer):
    """
    Initialises variational parameters to zero mean and identity covariance.
    Sensible initialisation for the final layer in a deep GP. This is also
    useful in testing, as starting KL divergences are exactly 0.
    """

    def __init__(self):
        super().__init__(q_sqrt_diagonal=1.0)
