# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
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

r"""
Kernels form a core component of GPflow models and allow prior information to
be encoded about a latent function of interest. The effect of choosing
different kernels, and how it is possible to combine multiple kernels is shown
in the `"Using kernels in GPflow" notebook <notebooks/kernels.html>`_.

Broadcasting over leading dimensions:
`kernel.K(X1, X2)` returns the kernel evaluated on every pair in X1 and X2.
E.g. if X1 has shape [S1, N1, D] and X2 has shape [S2, N2, D], kernel.K(X1, X2)
will return a tensor of shape [S1, N1, S2, N2]. Similarly, kernel.K(X1, X1)
returns a tensor of shape [S1, N1, S1, N1]. In contrast, the return shape of
kernel.K(X1) is [S1, N1, N1]. (Without leading dimensions, the behaviour of
kernel.K(X, None) is identical to kernel.K(X, X).)
"""

import abc
from functools import partial, reduce
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import AnyNDArray, Module, TensorType

ActiveDims = Union[slice, Sequence[int]]
NormalizedActiveDims = Union[slice, AnyNDArray]

class DistributionalKernel(Module, metaclass=abc.ABCMeta):

    """
    The basic distributional kernel class. Does not handle active dims or any other associated checks for ARD parameters.
    """

    def __init__(
        self, name: Optional[str] = None
    ) -> None:
        """
        :param name: optional kernel name.
        """
        super().__init__(name=name)

    @abc.abstractmethod
    def K(self, X: tfp.distributions.MultivariateNormalDiag, 
        X2: Optional[tfp.distributions.MultivariateNormalDiag] = None,
        ) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X: tfp.distributions.MultivariateNormalDiag) -> tf.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        X: tfp.distributions.MultivariateNormalDiag,
        X2: Optional[tfp.distributions.MultivariateNormalDiag] = None,
        *,
        full_cov: bool = True,
        presliced: bool = False,
        seed: int = None) -> tf.Tensor:
        
        if (not full_cov) and (X2 is not None):
            raise ValueError("Ambiguous inputs: `not full_cov` and `X2` are not compatible.")

        if not full_cov:
            assert X2 is None

            return self.K_diag(X)

        else:
            
            return self.K(X, X2, seed = seed)

    def __add__(self, other: "DistributionalKernel") -> "DistributionalKernel":
        return Sum([self, other])

    def __mul__(self, other: "DistributionalKernel") -> "DistributionalKernel":
        return Product([self, other])


class DistributionalCombination(DistributionalKernel):
    
    """
    Combine a list of distributional kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the distributional kernels to be combined are generated from their class
    names.
    """

    _reduction = None

    def __init__(self, kernels: Sequence[DistributionalKernel], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        if not all(isinstance(k, DistributionalKernel) for k in kernels):
            raise TypeError("can only combine DistributionalKernel instances")  # pragma: no cover

        self.kernels: List[DistributionalKernel] = []
        self._set_kernels(kernels)

    def _set_kernels(self, kernels: Sequence[DistributionalKernel]) -> None:
        # add kernels to a list, flattening out instances of this class therein
        kernels_list: List[DistributionalKernel] = []
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernels)
            else:
                kernels_list.append(k)
        self.kernels = kernels_list


class ReducingCombination(DistributionalCombination):
    def __call__(
        self,
        X: TensorType,
        X2: Optional[TensorType] = None,
        *,
        full_cov: bool = True
    ) -> tf.Tensor:

        return self._reduce(
            [k(X,  X2, full_cov=full_cov) for k in self.kernels]
        )

    def K(self, X: tfp.distributions.MultivariateNormalDiag, 
        X2: Optional[tfp.distributions.MultivariateNormalDiag] = None,
        ) -> tf.Tensor:
        return self._reduce([k.K(X, X2) for k in self.kernels])

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self._reduce([k.K_diag(X) for k in self.kernels])

    @property
    @abc.abstractmethod
    def _reduce(self) -> Callable[[Sequence[TensorType]], TensorType]:
        pass


class Sum(ReducingCombination):
    @property
    def _reduce(self) -> Callable[[Sequence[TensorType]], TensorType]:
        return tf.add_n  # type: ignore


class Product(ReducingCombination):
    @property
    def _reduce(self) -> Callable[[Sequence[TensorType]], TensorType]:
        return partial(reduce, tf.multiply)
