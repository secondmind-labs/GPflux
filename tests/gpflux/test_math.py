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
import numpy as np

from gpflux.math import compute_A_inv_b


def _get_psd_matrix(N):
    """Returns P.S.D matrix with shape [N, N]"""
    from gpflow.kernels import SquaredExponential

    x = np.linspace(-1, 1, N).reshape(-1, 1)
    A = SquaredExponential()(x, full_cov=True).numpy()  # [N, N]
    return A + 1e-6 * np.eye(N, dtype=A.dtype)


def test_compute_A_inv_x():
    N = 100
    A = _get_psd_matrix(N)
    b = np.random.randn(N, 2) / 100
    np.testing.assert_array_almost_equal(
        np.linalg.inv(A) @ b, compute_A_inv_b(A, b).numpy(), decimal=3,
    )
