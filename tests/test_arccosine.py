import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.test_util import session_tf
from sklearn.feature_extraction.image import extract_patches_2d

import gpflux
from gpflux.convolution.convolution_kernel import ImageRBF, ImageArcCosine, ImageStationary

padding = 'VALID'
strides = (1, 1, 1, 1)

def _J(order, theta):
    if order == 0:
        return np.pi - theta
    elif order == 1:
        return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
    elif order == 2:
        return (3.0 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (1.0 + 2.0 * tf.cos(theta)**2))
    else:
        raise NotImplementedError


def _image_to_patches(X, patch_shape):
    N = X.shape[0]
    patches = np.array([extract_patches_2d(im, patch_shape) for im in X])
    patches = np.reshape(patches, [-1, np.prod(patch_shape)])  # N*P x h*w
    return patches

class Data:
    N, H, W, C = image_shape = 20, 28, 28, 1
    M = 10
    w, h = patch_shape = [5, 5]
    X = np.random.randn(*image_shape)
    X_2d = np.reshape(X, [N, H*W*C])
    X_patches = _image_to_patches(X, patch_shape)
    Z = np.random.randn(M, np.prod(patch_shape))


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_Kdiag(session_tf, order, weight_variance, bias_variance, variance):
    def naive():
        patches = Data.X_patches
        kern = gpflow.kernels.ArcCosine(np.prod(Data.patch_shape),
                                        order=order, variance=variance,
                                        weight_variances=weight_variance,
                                        bias_variance=bias_variance, ARD=False)

        return kern.compute_Kdiag(patches).reshape(Data.N, -1)

    def conv():
        # X_patches_squared_norm = tf.matrix_diag_part(
                # gpflux.conv_square_dists.diag_conv_inner_prod(Data.X, Data.patch_shape)[..., 0])  # N x P
        ones = tf.ones((*Data.patch_shape, Data.C, 1), dtype=gpflow.settings.float_type)
        XpXpt = tf.nn.conv2d(Data.X ** 2, ones, strides, padding)
        X_patches_squared_norm = tf.reshape(XpXpt, (tf.shape(Data.X)[0], -1))  # N x P
        X_patches_squared_norm = weight_variance * X_patches_squared_norm + bias_variance
        theta = tf.cast(0, gpflow.settings.float_type)
        value = variance * (1. / np.pi) * _J(order, theta) * X_patches_squared_norm ** order
        return session_tf.run(value)

    expected = naive()
    value = conv()
    np.testing.assert_allclose(value, expected)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_Kuf(session_tf, order, weight_variance, bias_variance, variance):
    def naive():
        patches = Data.X_patches
        kern = gpflow.kernels.ArcCosine(np.prod(Data.patch_shape),
                                        order=order, variance=variance,
                                        weight_variances=weight_variance,
                                        bias_variance=bias_variance, ARD=False)

        return kern.compute_K(Data.Z, patches).reshape(Data.M, Data.N, -1)

    def conv():

        Z_filter = tf.transpose(tf.reshape(Data.Z, (Data.M, Data.h, Data.w, Data.C)), [1, 2, 3, 0])  # [h, w, C, M]
        XpZ = tf.nn.conv2d(Data.X, Z_filter, strides, padding)  # [N, Ph, Pw, M]
        ZXp = tf.reshape(tf.transpose(XpZ, [3, 0, 1, 2]), (Data.M, Data.N, -1))
        ZXp = weight_variance * ZXp + bias_variance  # [M, N, P]

        ZZt = tf.reduce_sum(Data.Z ** 2, axis=1)
        ZZt = tf.sqrt(weight_variance * ZZt + bias_variance)  # M

        ones = tf.ones((Data.h, Data.w, Data.C, 1), dtype=gpflow.settings.float_type)
        XpXpt = tf.nn.conv2d(Data.X ** 2, ones, strides, padding)
        XpXpt = tf.reshape(XpXpt, (Data.N, -1))
        XpXpt = tf.sqrt(weight_variance * XpXpt + bias_variance)  # [N, P]

        cos_theta = ZXp / ZZt[:, None, None] / XpXpt[None, :, :]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return session_tf.run(variance * (1. / np.pi) * _J(order, theta) *
                              ZZt[:, None, None] ** order *
                              XpXpt[None, :, :] ** order)
    expected = naive()  # [M, N, P]
    value = conv()  # [M, N, P]
    np.testing.assert_allclose(value, expected)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_ArcCosineImageKernel(session_tf, order, weight_variance, bias_variance, variance):

    patches = Data.X_patches
    kern = ImageArcCosine(image_shape=[28, 28, 1], patch_shape=[5, 5],
                       order=order, variance=variance,
                       weight_variances=weight_variance,
                       bias_variance=bias_variance, ARD=False)

    # Kdiag
    expected = kern.compute_Kdiag(patches).reshape(Data.N, -1)
    value = kern.compute_K_image(Data.X_2d)
    np.testing.assert_allclose(value, expected)

    # Kdiag [N, P, P]
    patches_per_image = np.reshape(patches, [Data.N, -1, np.prod(Data.patch_shape)])  # [N, P, h*w]
    expected = np.array([kern.compute_K_symm(x) for x in patches_per_image])  # [N, P, P]
    value = kern.compute_K_image_full_output_cov(Data.X_2d)
    np.testing.assert_allclose(value, expected)

    # Kuf
    expected = kern.compute_K(patches, Data.Z).reshape(Data.N, -1, Data.M)
    value = kern.compute_K_image_inducing_patches(Data.X_2d, Data.Z)
    np.testing.assert_allclose(value, expected)



@pytest.mark.parametrize("lengthscales", np.random.rand(2))
@pytest.mark.parametrize("variance", np.random.rand(2))
def test_RBFImageKernel(session_tf, lengthscales, variance):

    patches = Data.X_patches
    kern = ImageRBF(image_shape=[28, 28, 1], patch_shape=[5, 5],
                          variance=variance, lengthscales=lengthscales, ARD=False)

    # Kdiag [N,P]
    expected = kern.compute_Kdiag(patches).reshape(Data.N, -1)
    value = kern.compute_K_image(Data.X_2d)
    np.testing.assert_allclose(value, expected)

    # Kdiag [N, P, P]
    patches_per_image = np.reshape(patches, [Data.N, -1, np.prod(Data.patch_shape)])  # [N, P, h*w]
    expected = np.array([kern.compute_K_symm(x) for x in patches_per_image])  # [N, P, P]
    value = kern.compute_K_image_full_output_cov(Data.X_2d)
    np.testing.assert_allclose(value, expected)

    # Kuf
    expected = kern.compute_K(patches, Data.Z).reshape(Data.N, -1, Data.M)
    value = kern.compute_K_image_inducing_patches(Data.X_2d, Data.Z)
    np.testing.assert_allclose(value, expected)
