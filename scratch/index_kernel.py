import numpy as np
import gpflux
import gpflow

from data import mnist, mnist01

X, Y, Xs, Ys = mnist01()
print(X.shape)
print(Y.shape)
print(Xs.shape)
print(Ys.shape)

N = X.shape[0]
M = 100
H, W = 28, 28
assert H == W
patch_size = np.array([3, 3])
img_size_in = np.array([H, W])

base_kernel = gpflow.kernels.Matern12(np.prod(patch_size))
conv_kernel = gpflux.convolution_kernel.ConvKernel(base_kernel, img_size_in, patch_size)
index_kernel = gpflow.kernels.RBF(2)
kern = gpflux.convolution_kernel.IndexedConvKernel(conv_kernel, index_kernel)

inducing_patches = gpflux.inducing_patch.InducingPatch(np.random.randn(M, np.prod(patch_size)))
Z_indices = np.random.randint(0, H, size=(M, 2))
inducing_indices = gpflow.features.InducingPoints(Z_indices)
feat = gpflux.inducing_patch.IndexedInducingPatch(inducing_patches, inducing_indices)

like = gpflow.likelihoods.Gaussian()
m = gpflow.models.SVGP(X, Y, kern,
                       likelihood=like,
                       feat=feat,
                       num_latent=1,
                       minibatch_size=1000)

print(m.compute_log_likelihood())

from gpflow.training import AdamOptimizer
AdamOptimizer(0.01).minimize(m, maxiter=5)

print(m.compute_log_likelihood())

me, va = m.predict_f(Xs[:10, ...])

print(me.shape)
print(va.shape)

