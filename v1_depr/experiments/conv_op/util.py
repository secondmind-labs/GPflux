import pytest
import numpy as np
import tensorflow as tf
import gpflux
from gpflux.convolution import ConvPatchHandler, NaivePatchHandler


def create_op(op_type, image_shape, patch_shape, *args):
    op_types = {
        'conv': ConvPatchHandler,
        'naive': NaivePatchHandler
    }
    cfg = gpflux.convolution.ImagePatchConfig(image_shape, patch_shape)
    handler = op_types[op_type](cfg)
    return handler.image_patches_inner_product(*args)
