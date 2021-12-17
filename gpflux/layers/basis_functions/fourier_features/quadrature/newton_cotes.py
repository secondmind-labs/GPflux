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
"""
Kernel decompositon into features and coefficients based on Newton-Cotes quadrature.
"""
import tensorflow as tf

from gpflow.quadrature.gauss_hermite import repeat_as_list, reshape_Z_dZ


class SimpsonQuadratureFourierFeatures:
    def __init__(self, kernel, n_components, start=-1.0, stop=1.0):
        self.kernel = kernel

        self.n_components = n_components
        self.n_nodes = 2 * n_components + 1

        self.width = tf.truediv(stop - start, self.n_nodes - 1)
        self.abscissa, self.factor = transform(np.linspace(start, stop, self.n_nodes))

    def __call__(self, t):
        t_scaled = np.true_divide(t, self.kernel.lengthscales)
        weights = self._compute_weights(t_scaled)
        values = self._compute_values(t_scaled)
        return self.kernel.variance * np.sum(weights * values, axis=-1)

    @abstractmethod
    def _compute_weights(self, t):
        pass

    @abstractmethod
    def _compute_values(self, t):
        pass
