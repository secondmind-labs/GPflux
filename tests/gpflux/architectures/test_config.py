#
# Copyright (c) 2022 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Any

import pytest
import tensorflow_probability as tfp

from gpflux.architectures.config import (
    GaussianLikelihoodConfig,
    HeteroSkedasticLikelihoodConfig,
    LikelihoodConfig,
    StudenttLikelihoodConfig,
)


@pytest.fixture(
    name="likelihood_config",
    params=[
        GaussianLikelihoodConfig(noise_variance=1e-3),
        StudenttLikelihoodConfig(df=3, scale=1.0),
        HeteroSkedasticLikelihoodConfig(distribution_class=tfp.distributions.Normal),
        HeteroSkedasticLikelihoodConfig(distribution_class=tfp.distributions.StudentT),
    ],
)
def _likelihood_config(request: Any) -> LikelihoodConfig:
    return request.param


def test_likelihood_create(likelihood_config: LikelihoodConfig) -> None:
    try:
        likelihood_config.create()
    except:
        pytest.fail(f"Could not create likelihood with config: {type(likelihood_config)}")
