import numpy as np
import tensorflow as tf

import gpflow
import gpflow.multioutput.kernels as mk

from gpflow import params_as_tensors, params_as_tensors_for

float_type = gpflow.settings.float_type

class ConvKernel(mk.Mok):
    """
    Multi-output kernel for a GP from images to images. Constructed by
    convolving a patch function g(.) over the input image.
    """

    def __init__(self, basekern, img_size, patch_size, colour_channels=1):
        """
        :param basekern: gpflow.Kernel that operates on the vectors of length w*h
        :param img_size: tuple, W x H
        :param patch_size: tuple, w x h
        """
        gpflow.kernels.Kern.__init__(self, np.prod(img_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        # self.basekern.input_dim = np.prod(patch_size)  # Removed, b/c Conv shouldn't edit its arguments.
        self.colour_channels = colour_channels
        assert self.colour_channels == 1

    @gpflow.decors.params_as_tensors
    def _get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: N x (W*H*C)
        :return: Patches (N, num_patches, patch_size)
        """
        # castX = tf.transpose(
        #     tf.reshape(tf.cast(X, tf.float32, name="castX"), tf.stack([tf.shape(X)[0], -1, self.colour_channels])),
        #     [0, 2, 1])
        # castX = tf.cast(X, tf.float32, name="castX")

        # Roll the colour channel to the front, so it appears to `tf.extract_image_patches()` as separate images. Then
        # extract patches and reshape to have the first axis the same as the number of images. The separate patches will
        # then be in the second axis.
        castX = tf.transpose(tf.reshape(X, tf.stack([tf.shape(X)[0], -1, self.colour_channels])),
                             [0, 2, 1])
        castX = tf.cast(castX, tf.float32, name="castX")
        castX_r = tf.reshape(castX, [-1, self.img_size[0], self.img_size[1], 1], name="rX")
        patches = tf.extract_image_patches(castX_r,
                                           [1, self.patch_size[0], self.patch_size[1], 1],
                                           [1, 1, 1, 1],
                                           [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        patches_r = tf.reshape(patches, [tf.shape(X)[0], self.colour_channels * shp[1] * shp[2], shp[3]])
        return tf.cast(patches_r, gpflow.settings.tf_float)

    @gpflow.params_as_tensors
    def K(self, X, X2=None, full_output_cov=False):
        Xp = self._get_patches(X)  # N x P x wh
        N = tf.shape(Xp)[0]
        P = tf.shape(Xp)[1]

        if full_output_cov:
            Xp2 = tf.reshape(self._get_patches(X2), (-1, self.patch_len)) if X2 is not None else None
            N2 = tf.shape(Xp2)[0] if Xp2 is not None else N
            Xp = tf.reshape(Xp, (N * P, self.patch_len))  # NP x wh
            return tf.reshape(self.basekern.K(Xp, Xp2), (N, P, N2, P))  # N x P x N2 x P
        else:
            Xp_t = tf.transpose(Xp, [1, 0, 2])  # P x N x wh
            if X2 is not None:
                Xp2_t = tf.transpose(self._get_patches(X2), [1, 0, 2])  # P x N x wh
                return tf.map_fn(lambda Xs: self.basekern.K(*Xs) , (Xp_t, Xp2_t), float_type)  # P x N x N2
            else:
                return tf.map_fn(lambda X: self.basekern.K(X) , Xp_t, float_type)  # P x N x N


    @gpflow.params_as_tensors
    def Kdiag(self, X, full_output_cov=False):
        Xp = self._get_patches(X)  # N x P x wh

        if full_output_cov:
            return tf.map_fn(lambda x: self.basekern.K(x), Xp)  # N x P x P
        else:
            return tf.map_fn(lambda x: self.basekern.Kdiag(x), Xp)  # N x P

    ## Properties
    ## --------------------------

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return ((self.img_size[0] - self.patch_size[0] + 1) *
                (self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels)

    @property
    def num_outputs(self):
        return self.num_patches

    @property
    def Hout(self):
        return self.img_size[0] - self.patch_size[0] + 1

    @property
    def Wout(self):
        return self.img_size[1] - self.patch_size[1] + 1

    ## Autoflow
    ## --------------------------

    @gpflow.autoflow((gpflow.settings.tf_float,))
    def compute_patches(self, X):
        return self._get_patches(X)


class IndexedConvKernel:

    def __init__(self, conv_kernel, index_kernel):
        self.conv_kernel = conv_kernel
        self.index_kernel = index_kernel
