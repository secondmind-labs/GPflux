# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
from typing import Optional

from gpflow.inducing_variables import InducingPoints

from gpflux.initializers.variational import (
    VariationalInitializer,
    MeanFieldVariationalInitializer,
)


class Initializer(ABC):
    """Base object that initialises variational parameters and inducing points"""

    def __init__(
        self,
        init_at_predict: bool,
        qu_initializer: Optional[VariationalInitializer] = None,
    ):
        self.init_at_predict = init_at_predict
        if qu_initializer is None:
            qu_initializer = MeanFieldVariationalInitializer()
        self.qu_initializer = qu_initializer

    def init_variational_params(self, q_mu, q_sqrt) -> None:
        self.qu_initializer.init_variational_params(q_mu, q_sqrt)

    def init_inducing_variable(self, inducing_variable, inputs=None) -> None:
        # HACK to deal with multioutput inducing variables
        if hasattr(inducing_variable, "inducing_variable_list"):
            inducing_variable_list = inducing_variable.inducing_variable_list
        elif hasattr(inducing_variable, "inducing_variable_shared"):
            inducing_variable_list = [inducing_variable.inducing_variable_shared]
        else:
            raise AttributeError("Could not find inducing variable attribute")

        for inducing_var in inducing_variable_list:
            if self.init_at_predict:
                self.init_single_inducing_variable(inducing_var, inputs=inputs)
            else:
                self.init_single_inducing_variable(inducing_var)

    @abstractmethod
    def init_single_inducing_variable(
        self, inducing_variable: InducingPoints, inputs=None
    ) -> None:
        """
        Initializes the inducing variable (here assumed to be InducingPoints)
        for a single GP.
        Should only use `inputs` when self.init_at_predict is True.
        """
        raise NotImplementedError
