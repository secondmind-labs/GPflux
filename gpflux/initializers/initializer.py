# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from abc import ABC, abstractmethod
from typing import Optional

from gpflow.base import TensorType
from gpflow.inducing_variables import InducingPoints, InducingVariables


class Initializer(ABC):
    """Base object that initialises variational parameters and inducing points"""

    def __init__(
        self, init_at_predict: bool,
    ):
        self.init_at_predict = init_at_predict

    def init_inducing_variable(
        self, inducing_variable: InducingVariables, inputs: Optional[TensorType] = None
    ) -> None:
        for inducing_var in inducing_variable.inducing_variables:
            if self.init_at_predict:
                self.init_single_inducing_variable(inducing_var, inputs=inputs)
            else:
                self.init_single_inducing_variable(inducing_var)

    @abstractmethod
    def init_single_inducing_variable(
        self, inducing_variable: InducingPoints, inputs: Optional[TensorType] = None
    ) -> None:
        """
        Initializes the inducing variable (here assumed to be InducingPoints)
        for a single GP.
        Should only use `inputs` when self.init_at_predict is True.
        """
        raise NotImplementedError
