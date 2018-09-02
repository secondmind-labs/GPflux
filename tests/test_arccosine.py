import numpy as np
import pytest
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches_2d

import gpflow
import gpflux
from gpflow.test_util import session_tf

padding = 'VALID'
strides = (1, 1, 1, 1)

def _J(order, theta):
    if order == 0:
        return np.pi - theta
    elif order == 1:
        return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
    elif order == 2:
        return (3.0 * tf.sin(theta) * tf.cos(theta) + 
                (np.pi - theta) * (1.0 + 2.0 * tf.cos(theta)**2))
    else:
        raise NotImplementedError


def _image_to_patches(X, patch_size):
    N = X.shape[0]
    patches = np.array([extract_patches_2d(im, patch_size) for im in X])
    patches = np.reshape(patches, [-1, np.prod(patch_size)])  # N*P x h*w
    return patches

class Data:
    N, H, W, C = image_shape = 20, 28, 28, 1
    M = 10
    w, h = patch_size = [5, 5]
    X = np.random.randn(*image_shape)
    X_2d = np.reshape(X, [N, H*W*C])
    X_patches = _image_to_patches(X, patch_size)
    Z = np.random.randn(M, np.prod(patch_size))


# @pytest.mark.skip
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_Kdiag(session_tf, order, weight_variance, bias_variance, variance):
    def naive():
        patches = Data.X_patches
        kern = gpflow.kernels.ArcCosine(np.prod(Data.patch_size), 
                                        order=order, variance=variance,
                                        weight_variances=weight_variance,
                                        bias_variance=bias_variance, ARD=False)

        return kern.compute_Kdiag(patches).reshape(Data.N, -1)

    def conv():
        # X_patches_squared_norm = tf.matrix_diag_part(
                # gpflux.conv_square_dists.diag_conv_inner_prod(Data.X, Data.patch_size)[..., 0])  # N x P
        ones = tf.ones((*Data.patch_size, Data.C, 1), dtype=gpflow.settings.float_type)
        XpXpt = tf.nn.conv2d(Data.X ** 2, ones, strides, padding)
        X_patches_squared_norm = tf.reshape(XpXpt, (tf.shape(Data.X)[0], -1))  # N x P
        X_patches_squared_norm = weight_variance * X_patches_squared_norm + bias_variance
        theta = tf.cast(0, gpflow.settings.float_type)
        value = variance * (1. / np.pi) * _J(order, theta) * X_patches_squared_norm ** order
        return session_tf.run(value)

    expected = naive()
    value = conv()
    np.testing.assert_allclose(value, expected)


# @pytest.mark.skip
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_Kuf(session_tf, order, weight_variance, bias_variance, variance):
    def naive():
        patches = Data.X_patches
        kern = gpflow.kernels.ArcCosine(np.prod(Data.patch_size), 
                                        order=order, variance=variance,
                                        weight_variances=weight_variance,
                                        bias_variance=bias_variance, ARD=False)

        return kern.compute_K(Data.Z, patches).reshape(Data.M, Data.N, -1)

    def conv():

        Z_filter = tf.transpose(tf.reshape(Data.Z, (Data.M, Data.h, Data.w, Data.C)), [1, 2, 3, 0])  # [h, w, C, M]
        XpZ = tf.nn.conv2d(Data.X, Z_filter, strides, padding)  # [N, Ph, Pw, M]
        ZXp = tf.reshape(tf.transpose(XpZ, [3, 0, 1, 2]), (Data.M, Data.N, -1))
        ZXp = weight_variance * ZXp + bias_variance  # M x N x P

        ZZt = tf.reduce_sum(Data.Z ** 2, axis=1)
        ZZt = tf.sqrt(weight_variance * ZZt + bias_variance)  # M

        ones = tf.ones((Data.h, Data.w, Data.C, 1), dtype=gpflow.settings.float_type)
        XpXpt = tf.nn.conv2d(Data.X ** 2, ones, strides, padding)
        XpXpt = tf.reshape(XpXpt, (Data.N, -1))
        XpXpt = tf.sqrt(weight_variance * XpXpt + bias_variance)  # N x P

        cos_theta = ZXp / ZZt[:, None, None] / XpXpt[None, :, :]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return session_tf.run(variance * (1. / np.pi) * _J(order, theta) *
                              ZZt[:, None, None] ** order *
                              XpXpt[None, :, :] ** order)
    
    expected = naive()  # M x N x P
    value = conv()  # M x N x P
    
    np.testing.assert_allclose(value, expected)

from gpflux.convolution.convolution_kernel import StationaryImageKernel, ArcCosineImageKernel

@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("weight_variance", np.random.rand(1))
@pytest.mark.parametrize("bias_variance", np.random.rand(1))
@pytest.mark.parametrize("variance", np.random.rand(1))
def test_ArcCosineImageKernel(session_tf, order, weight_variance, bias_variance, variance):

    patches = Data.X_patches
    kern = ArcCosineImageKernel(np.prod(Data.patch_size), 
                                image_size=[28, 28, 1], patch_size=[5, 5],
                                order=order, variance=variance,
                                weight_variances=weight_variance,
                                bias_variance=bias_variance, ARD=False)

    # Kdiag
    expected = kern.compute_Kdiag(patches).reshape(Data.N, -1)
    value = kern.compute_K_image(Data.X_2d)
    np.testing.assert_allclose(value, expected)

    # Kuf
    expected = kern.compute_K(Data.Z, patches).reshape(Data.M, Data.N, -1)
    value = kern.compute_K_image_inducing_patches(Data.X_2d, Data.Z)
    np.testing.assert_allclose(value, expected)
