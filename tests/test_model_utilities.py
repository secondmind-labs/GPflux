# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np
import gpflow
import gpflux

from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.features import InducingPoints
from gpflow.training import AdamOptimizer

class Data:
    M = 100
    N, D = 1000, 28**2
    X = np.random.randn(N , D)
    LATENT_DIM = 2
    OUTPUT_DIMS = [50, 30**2, D]
    PATCH_SIZE = [3, 3]
    BATCH_SIZE = 50

def test_to_str():
    """
    This test build a deep latent GP model consisting of 3 layers:
    SVGP, Linear and Deconv Layer and check if __str__
    functionality works as desired
    """

    latent_layer = gpflux.layers.LatentVariableConcatLayer(Data.LATENT_DIM, Data.D)

    ### Decoder
    Z1 = np.random.randn(Data.M, Data.LATENT_DIM)
    feat1 = gpflow.features.InducingPoints(Z1)
    kern1 = gpflow.kernels.RBF(Data.LATENT_DIM, lengthscales=float(Data.LATENT_DIM) ** 0.5)
    layer1 = gpflux.layers.GPLayer(kern1, feat1, Data.OUTPUT_DIMS[0])

    # Layer 2: Linear
    layer2 = gpflux.layers.LinearLayer(Data.OUTPUT_DIMS[0], Data.OUTPUT_DIMS[1])

    # Layer 3
    patch_init = gpflux.init.PatchSamplerInitializer(Data.X[:10], 28, 28)
    layer3 = gpflux.layers.ConvLayer([30, 30], [28, 28], Data.M, [3, 3],
                                     patches_initializer=patch_init)

    model = gpflux.DeepGP(np.empty((Data.N, 0)),
                          Data.X,
                          layers=[latent_layer, layer1, layer2, layer3])

    description = model.describe()
    description = description.split('\n')
    assert len(description) == 9
    assert description[0] == "DeepGP"
    assert "LatentVariable" in description[3] \
            and str(Data.LATENT_DIM) in description[3]
    assert "GPLayer" in description[4]
    assert "LinearLayer" in description[5]
    assert "GPLayer" in description[6]
    assert "Gaussian" in description[8]
