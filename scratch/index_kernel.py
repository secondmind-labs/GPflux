import numpy as np
import gpflux
import gpflow

M = 5
N = 10
patch_size = [3, 3]
img_size = [28, 28]

X = np.random.randn(N, (img_size[0] + patch_size[0] - 1)**2)
Y = np.random.randn(N, img_size[0]**2)

base_kernel = gpflow.kernels.Matern12(np.prod(patch_size))
conv_kernel = gpflux.convolution_kernel.ConvKernel(base_kernel, img_size, patch_size)
index_kernel = gpflow.kernels.RBF(2)
kern = gpflux.convolution_kernel.IndexedConvKernel(conv_kernel, index_kernel)

inducing_patches = gpflux.inducing_patch.InducingPatch(np.random.randn(M, np.prod(patch_size)))
inducing_indices = 
