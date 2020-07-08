from typing import Tuple, Optional

import numpy as np

from gpflow.models.training_mixins import RegressionData
from gpflux.vish.spherical_harmonics import num_harmonics


def get_num_inducing(kernel_type: str, dimension: int, max_degree: int) -> int:
    """
    Returns the number of basis functions in the approximation.
    """
    assert kernel_type in ["matern", "arccosine"]

    if kernel_type == "matern":
        # For the Matern class kernels all the levels (or degrees) are used.
        return sum(num_harmonics(dimension, d) for d in range(max_degree))
    elif kernel_type == "arccosine":
        harmonics_per_level = [num_harmonics(dimension, d) for d in range(max_degree)]
        # For the ArcCosine kernel the eigenvalues for all the odd levels larger than 2 or zero
        # We sum the number of harmonics at the following level 0, 1, 2, 4, 6, 8, ...
        return sum(harmonics_per_level[:3]) + sum(harmonics_per_level[4::2])
    else:
        raise NotImplementedError


def get_max_degree_closest_but_smaller_than_num_inducing(
    kernel_type, dimension: int, num_inducing: int
) -> Optional[Tuple[int, int]]:
    r"""
    Returns the max degree such that the total number of harmonics for the given
    `dimension` is the closest but smaller or equal than `num_inducing`.

    For matern kernels this corresponds to
    \sum_{degree=0}^{max_degree-1} N(dimension, degree) <= num_inducing (1)
    with max_degree the largest int such that (1) is true.

    For ArcCosine kenrel this corresponds degree is 0, 1, 2 or an odd number.

    Returns the degree and the number of inducing
    """
    assert kernel_type in ["arccosine", "matern"]
    num_inducing_per_degree = []
    max_degrees = []
    for max_degree in range(100):
        num_inducing_per_degree.append(
            get_num_inducing(kernel_type, dimension, max_degree)
        )
        max_degrees.append(max_degree)
        if num_inducing_per_degree[-1] > num_inducing:
            return max_degrees[-2], num_inducing_per_degree[-2]

    # Nothing found
    return None


def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add a bias of 1 to each point (row) in `X`.
        If X.shape[1] == 1, a double bias is added
        so that the dimensionality of X is at least 3D.

    :param X: [N, D]
    :return: [N, max(3, D+1)]
    """
    num_data, input_dim = X.shape
    num_bias_dimensions = max(3 - input_dim, 1)
    return np.c_[X, np.ones((num_data, num_bias_dimensions), dtype=X.dtype)]  # [N, D+1]


def range_scaler(X: np.ndarray, r=1.0):
    """
    Scales `X` so that for each column of `X` all values fall
    between [-r and r].
    """
    X_max = X.max(axis=0, keepdims=True)
    X_min = X.min(axis=0, keepdims=True)
    spread = X_max - X_min  # [N, D]
    X = (X - X_min) / spread  # [N, D], value range [0, 1]
    X = 2.0 * r * (X - 0.5)  # [N, D], range [-r, r]
    return X


def preprocess_data(
    data: RegressionData,
) -> Tuple[RegressionData, np.ndarray, np.ndarray]:
    """
    Takes the raw regression points data = (X, Y) and does the following operations:
    - Normalise Y to have zero mean and one standard dev.
    - Rescales each column of X to lie between -1 and 1.
    - Add a bias of 1 to each point in X. If X.shape[1] == 1, a double bias is added
        so that the dimensionality of X is at least 3D.

    Also returns the mean and std dev of Y.

    :param data: pair of X and Y
    :return: (X, Y), Y_mean, Y_std
    """
    X, Y = data

    # Normalise Y
    Y_mean = Y.mean(axis=0, keepdims=True)  # [N, 1]
    Y_std = Y.std(axis=0, keepdims=True)  # [N, 1]
    Y_rescaled = (Y - Y_mean) / Y_std  # [N, 1]

    # Rescale X to be within [-1, 1]
    X = range_scaler(X)

    # Add bias to X
    X_with_bias = add_bias(X)

    # Checks
    assert X_with_bias.shape[1] >= 3, "Data needs to be at least 3D"
    assert X_with_bias.shape[0] == Y_rescaled.shape[0]
    assert Y_rescaled.shape[1] == 1

    return (X_with_bias, Y_rescaled), Y_mean, Y_std
