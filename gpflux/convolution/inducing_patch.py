# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np

from gpflow import Param, settings
from gpflow.multioutput.features import Mof


class InducingPatch(Mof):
    """
    Inducing features which are typically used in combination with
    convolutional kernels.
    """

    def __init__(self, Z):
        """
        :param Z: np.array
            shape: M x w x h or M x wh
        """
        super().__init__()
        if Z.ndim == 3:
            M, w, h = Z.shape
            Z = np.reshape(Z, [M, w * h])  # M x wh

        self.Z = Param(Z, dtype=settings.float_type)  # M x wh

    def __len__(self):
        return self.Z.shape[0]

    @property
    def outputs(self):  # a.k.a. L
        return 1


class IndexedInducingPatch(InducingPatch):
    """
    Inducing feature combining patches and indices.
    """

    def __init__(self, patches, indices):
        super().__init__(patches)
        self.indices = Param(indices / indices.max(axis=0))

    @property
    def patches(self):
        return self.Z
