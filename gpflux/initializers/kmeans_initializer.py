# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

from .z_initializer import GivenZInitializer


def init_inducing_points(X: np.ndarray, num_inducing: int) -> np.ndarray:
    num_data, input_dim = X.shape
    if num_data > num_inducing:
        centroids, labels = kmeans2(X, k=num_inducing, minit="points")
        assert set(labels) == set(range(num_inducing))
        return centroids
    else:
        warnings.warn("Requesting more inducing points than data set size is inefficient!")
        # if we have less data points than number of inducing points requested,
        # add randomly generated points (implicitly assumes data is zero-one
        # normalized):
        extra_rows = np.random.randn(num_inducing - num_data, input_dim)
        return np.concatenate([X, extra_rows], axis=0)


class KmeansInitializer(GivenZInitializer):
    """
    Initializes inducing points to k-means clustering of the dataset.
    """

    def __init__(
        self, X: np.ndarray, num_inducing: int,
    ):
        """
        :param X: dataset whose k-means clusters to use for inducing points.
        :param num_inducing: number of clusters to select
        """
        Z = init_inducing_points(X, num_inducing)
        super().__init__(Z=Z)
