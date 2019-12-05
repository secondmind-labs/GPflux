# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from abc import ABC, abstractmethod

class Initializer(ABC):
    """Base object that initialises variational parameters and inducing points"""

    def __init__(self):
        super().__init__()
        self.deferred_init = False

    @abstractmethod
    def init_variational_params(self, q_mu, q_sqrt) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_inducing_variable(self, inducing_variable, input_data=None) -> None:
        raise NotImplementedError

