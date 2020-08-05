import numpy as np


from gpflux.sampling.utils import compute_A_inv_b


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
