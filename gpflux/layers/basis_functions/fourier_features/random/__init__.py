#
# Copyright (c) 2021 The GPflux Contributors.
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
""" A kernel's features and coefficients using Random Fourier Features (RFF). """

from gpflux.layers.basis_functions.fourier_features.random.base import (
    RandomFourierFeatures,
    RandomFourierFeaturesCosine,
)
from gpflux.layers.basis_functions.fourier_features.random.orthogonal import (
    OrthogonalRandomFeatures,
)

__all__ = [
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
    "RandomFourierFeaturesCosine",
]
