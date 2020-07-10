# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import Optional
import warnings
import numpy as np
import tensorflow as tf

from gpflow.inducing_variables import InducingPoints

from .initializer import Initializer


class GivenZInitializer(Initializer):
    """
    Initialises the inducing points to a given value.
    """

    def __init__(
        self, Z: Optional[np.ndarray] = None,
    ):
        """
        :param Z: manually specified inducing point locations
            if None, inducing points will be initialized to zeros
        """
        super().__init__(init_at_predict=False)
        self.Z = Z

    def init_single_inducing_variable(
        self, inducing_variable: InducingPoints, inputs=None
    ) -> None:
        if self.Z is None:
            warnings.warn("No Z specified, initializing inducing_variable to zeros!")
            Z = tf.zeros_like(inducing_variable.Z)
        else:
            Z = self.Z
        inducing_variable.Z.assign(Z)
