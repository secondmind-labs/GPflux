# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Any, Optional, Type

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import TensorType
from gpflow.likelihoods import MultiLatentTFPConditional
from gpflow.utilities import positive


class HeteroskedasticTFPConditional(MultiLatentTFPConditional):
    """
    Heteroskedastic Likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    The `loc` and `scale` of the distribution are given by a
    two-dimensional multi-output GP.

    #NOTE -- inserted option to provide degrees of freedom specifically for Student-t lik.
    """

    def __init__(
        self,
        distribution_class: Type[tfp.distributions.Distribution] = tfp.distributions.StudentT,
        scale_transform: Optional[tfp.bijectors.Bijector] = None,
        df=4,
        **kwargs: Any,
    ) -> None:
        """
        :param distribution_class: distribution class parameterized by `loc` and `scale`
            as first and second argument, respectively.
        :param scale_transform: callable/bijector applied to the latent
            function modelling the scale to ensure its positivity.
            Typically, `tf.exp` or `tf.softplus`, but can be any function f: R -> R^+.
            Defaults to exp if not explicitly specified.
        """
        if scale_transform is None:
            scale_transform = positive(base="exp")
        self.scale_transform = scale_transform
        self.df = df

        def conditional_distribution(Fs: TensorType) -> tfp.distributions.Distribution:
            tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
            loc = Fs[..., :1]
            scale = self.scale_transform(Fs[..., 1:])
            return distribution_class(self.df, loc, scale)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )
