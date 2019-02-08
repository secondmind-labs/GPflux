# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np
import tensorflow as tf

import gpflow
from gpflow import autoflow, settings, Param
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import MeanFunction


class IdentityConvMean(MeanFunction):
    def __init__(self, image_shape, filter_shape, feature_maps_out=1, stride=1):
        super().__init__()
        if len(image_shape) == 2:
            # image_shape only contains [H, W],
            # so we assume a one-dimensional color channel
            image_shape += [1]
        
        self.image_shape = image_shape
        self.filter_shape = filter_shape if len(filter_shape) == 1 else filter_shape[0]
        self.feature_maps_in = image_shape[-1]
        self.feature_maps_out = feature_maps_out
        self.stride = stride
        self.conv_filter = self._init_filter()

    @params_as_tensors
    def __call__(self, X):
        # reshape to image of shape [N, W, H, C]
        N = tf.shape(X)[0]
        NHWC_X = tf.reshape(X, (N, *self.image_shape))

        # apply conv operation
        NHWC_m = tf.nn.conv2d(
            NHWC_X,
            self.conv_filter,
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
            data_format="NHWC"
        )

        # reshape to flattened representation
        return tf.reshape(NHWC_m, [N, -1])

    def _init_filter(self):
        filter_shape = (
            self.filter_shape,
            self.filter_shape,
            self.feature_maps_in,
            self.feature_maps_out
        )
        identity_filter = np.zeros(filter_shape, dtype=settings.float_type)
        identity_filter[self.filter_shape // 2, self.filter_shape // 2, :, :] = 1.0
        return identity_filter
    
    @autoflow((settings.float_type, [None, None]))
    def compute(self, X):
        return self.__call__(X)
    
    def compute_default_graph(self, X):
        return tf.Session().run(self.__call__(tf.convert_to_tensor(X)))
