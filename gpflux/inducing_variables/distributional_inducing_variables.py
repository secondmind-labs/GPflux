# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated

from gpflow.base import Module, Parameter, TensorData, TensorType
from gpflow.utilities import positive


class DistributionalInducingVariables(Module):
    """
    Abstract base class for distributional inducing variables.
    """

    @property
    @abc.abstractmethod
    def num_inducing(self) -> tf.Tensor:
        """
        Returns the number of inducing variables, relevant for example to determine the size of the
        variational distribution.
        """
        raise NotImplementedError

    @deprecated(
        reason="len(iv) should return an `int`, but this actually returns a `tf.Tensor`."
        " Use `iv.num_inducing` instead."
    )
    def __len__(self) -> tf.Tensor:
        return self.num_inducing


class DistributionalInducingPointsBase(DistributionalInducingVariables):
    def __init__(self, Z_mean: TensorData, Z_var: TensorData, name: Optional[str] = None):
        """
        :param Z_mean: the initial mean positions of the inducing points, size [M, D]
        :param Z_var: the initial var positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z_mean, (tf.Variable, tfp.util.TransformedVariable)):
            Z_mean = Parameter(Z_mean, name = f"{self.name}_Z_mean" if self.name else "Z_mean")
        self.Z_mean = Z_mean

        if not isinstance(Z_var, (tf.Variable, tfp.util.TransformedVariable)):
            Z_var = Parameter(value=Z_var, transform=positive(), name = f"{self.name}_Z_var" if self.name else "Z_var")
        self.Z_var = Z_var

    @property
    def num_inducing(self) -> Optional[tf.Tensor]:
        return tf.shape(self.Z_mean)[0]


class DistributionalInducingPoints(DistributionalInducingPointsBase):
    """
    Real-space inducing points
    """

