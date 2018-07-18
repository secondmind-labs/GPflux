import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings
from gpflow.multioutput.features import Mof

logger = settings.logger()

class InducingPatch(Mof):
    def __init__(self, Z):
        """
        :param Z: np.array
            shape: M x w x h or M x wh
        """
        super().__init__()
        if Z.ndim == 3:
            M, w, h = Z.shape
            Z = np.reshape(Z, [M, w * h])  # M x wh

        print("inducing patches shape", Z.shape)
        self.Z = gpflow.Param(Z, dtype=gpflow.settings.float_type)  # M x wh

    def __len__(self):
        return self.Z.shape[0]

    @property
    def outputs(self):  # a.k.a. L
        return 1


class IndexedInducingPatch:
    def __init__(self, inducing_patch, inducing_indices):
        self.inducing_patch = inducing_patch
        self.inducing_indices = inducing_indices
