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

def test_deep_deconv_gp_setup_and_minimization():
    """
    This test build a deep GP model consisting of 3 layers:
    SVGP, Linear and Deconv Layer and checks if the model
    can be optimized.
    """

    enc = gpflux.encoders.RecognitionNetwork(Data.LATENT_DIM, Data.D, [256, 256])
    latent_layer = gpflux.layers.LatentVariableConcatLayer(Data.LATENT_DIM, encoder=enc)

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
                                     inducing_patches_initializer=patch_init)


    model = gpflux.DeepGP(np.empty((Data.N, 0)),
                          Data.X,
                          layers=[latent_layer, layer1, layer2, layer3])

    # minimize
    likelihood_before_opt = model.compute_log_likelihood()
    AdamOptimizer(0.01).minimize(model, maxiter=1)
    likelihood_after_opt = model.compute_log_likelihood()

    assert likelihood_before_opt < likelihood_after_opt

if __name__ == "__main__":
    test_deep_deconv_gp_setup_and_minimization()
