# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import gpflow
import numpy as np

from gpflow.kernels import RBF

from sklearn.feature_extraction.image import extract_patches_2d


class Initializer:
    """
    Base class for parameter initializers.
    It should be subclassed when implementing new types.
    Can be used for weights of a neural network, inducing patches, points, etc.
    """
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()  # pragma: no cover


class PatchSamplerInitializer(Initializer):

    def __init__(self, X, width=None, height=None, unique=False):
        """
        :param X: np.array
            N x W x H
        """
        if width is None and height is None:
            if X.ndim <= 2:
                raise ValueError("Impossible to infer image width and height")
            else:
                assert X.ndim == 3
                width, height = X.shape[1:]

        self.X = np.reshape(X, [-1, width, height])
        self.unique = unique

    def sample(self, shape):
        """
        :param shape: tuple
            M x w x h, number of patches x patch width x patch height
        :return: np.array
            returns M patches of size w x h, specified by the `shape` param.
        """
        num = shape[0]  # M
        patch_size = shape[1:]  # w x h

        patches = np.array([extract_patches_2d(im, patch_size) for im in self.X])
        patches = np.concatenate(patches, axis=0)
        patches = np.reshape(patches, [-1, np.prod(patch_size)])
        if self.unique:
            print("selecting unique patches")
            patches = np.unique(patches, axis=0)

        # patches = np.reshape(patches, [-1, *patch_size])  # (N * P) x w x h
        idx = np.random.permutation(range(len(patches)))[:num]  # M
        return patches[idx, ...].reshape(shape)  # M x w x h


class PatchIndexSamplerInitializer(PatchSamplerInitializer):

    def sample(self, shape, jitter=0.1):
        """
        Returns both the indices of the patches and the patches,
        jitter is added to indices
        :param shape: tuple
            M x w x h, number of patches x patch width x patch height
        :return: np.array
            returns M patches of size w x h, specified by the `shape` param.
        """
        h, w = shape[1:]
        val = self.X.shape[1] - shape[1] + 1
        indices = np.random.randint(3, val-3, size=[shape[0], 2])

        patches = []
        for i,j in indices:
            patch_added = False
            while not patch_added:
                random_index = np.random.randint(0, len(self.X))
                patch = self.X[random_index, i:i+h, j:j+h]
                if not np.array([(patch == p).all() for p in patches]).any():
                    patches.append(patch)
                    patch_added = True

        indices = indices.astype(np.float64) + np.random.randn(*indices.shape) * jitter
        return indices, np.array(patches)



class NormalInitializer(Initializer):
    """
    Sample initial weights from the Normal distribution.
    :param std: float
        Std of initial parameters.
    :param mean: float
        Mean of initial parameters.
    """
    def __init__(self, std=1.0, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)


class KernelStructureMixingMatrixInitializer(Initializer):
    """
    Initialization routine for the Mixing Matrix P,
    used in f(x) = P g(x).
    """

    def __init__(self, kern=None):
        self.kern = RBF(2, variance=2.0) if kern is None else kern

    def sample(self, shape):
        """
        :param shape: tuple, P x L.
        Note that P is both used for the dimension and the matrix.
        """
        #TODO(vincent): why is this image *width*? not width * height?
        im_width, num_latent = shape  # P x L
        IJ = np.vstack([x.flatten() for x in np.meshgrid(np.arange(im_width), np.arange(im_width))]).T
        K_IJ = self.kern.compute_K_symm(IJ) + np.eye(im_width ** 2) * 1E-6
        u, s, v = np.linalg.svd(K_IJ)
        P = (u[:, :num_latent] * s[None, :num_latent] ** 0.5)  # P x L
        return P


