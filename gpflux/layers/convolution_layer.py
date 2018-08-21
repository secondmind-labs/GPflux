# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import numpy as np
import gpflow

from typing import List, Optional, Union
from gpflow import settings

from .layers import GPLayer
from .. import init
from ..convolution import ConvKernel, InducingPatch, \
                    IndexedConvKernel, IndexedInducingPatch, \
                    PoolingIndexedConvKernel


def _check_input_output_shape(input_shape, output_shape, patch_size):
    width_check = (input_shape[0] - patch_size[0] + 1 == output_shape[0])
    height_check = (input_shape[1] - patch_size[1] + 1 == output_shape[1])
    return width_check and height_check

def _from_patches_initializer_to_patches(initializer, shape):
    """
    If initializer is an instance of init.Initializer it will create
    the patches by calling the initializer with the given shape.
    If the initializer is actually a np.ndarray the array gives
    the patches.
    """
    if isinstance(initializer, init.Initializer):
        return initializer(shape)  # M x w x h
    elif isinstance(initializer, np.ndarray):
        return initializer  # M x w x h
    else:
        raise ValueError



class ConvLayer(GPLayer):

    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 number_inducing: int,
                 patch_size: List, *,
                 stride: int = 1,
                 num_filters: int = 1,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
                 base_kernel_class: type = gpflow.kernels.RBF,
                 patches_initializer: Optional[Union[np.ndarray, init.Initializer]] \
                         = init.NormalInitializer()):
        """
        This layer constructs a convolutional GP layer.
        :input_shape: tuple
            shape of the input images, W x H
        :output_shape: tuple
            shape of the output images
        :param patch_size: tuple
            Shape of the patches (a.k.a kernel_size of filter_size)
        :param number_inducing: int
            Number of inducing patches, M

        Optional:
        :param stride: int
            An integer specifying the strides of the convolution along the height and width.
        :param num_filters: int
            Number of filters in the convolution
        :param q_mu and q_sqrt: np.ndarrays
            Variatial posterior parameterisation.
        :param patches_initializer: init.Initializer or np.ndarray
            Instance of the class `init.Initializer` that initializes the inducing patches,
            can also be a np.ndarray, if this is the case the patches_initializer param
            holds the inducing patches M x w x h
        """
        raise NotImplementedError("Convolutional Layers are deprecated for the time being")

        assert num_filters == 1 and stride == 1  # TODO

        if not _check_input_output_shape(input_shape, output_shape, patch_size):
            print("input_shape: ", input_shape)
            print("output_shape: ", output_shape)
            print("patch_size: ", patch_size)
            raise ValueError("The input, output and patch size are inconsistent in the ConvLayer. "
                             "The correct dimension should be: output = input - patch_size + 1.")

        # inducing patches
        shape = [number_inducing, *patch_size]  # tuple with values: M x w x h
        init_patches = _from_patches_initializer_to_patches(patches_initializer, shape)  # M x w x h
        inducing_patches = InducingPatch(init_patches)

        base_kernel = base_kernel_class(np.prod(patch_size))  # TODO: we are using the default kernel hyps
        conv_kernel = ConvKernel(base_kernel, input_shape, patch_size, colour_channels=1)  # TODO add colour

        super().__init__(conv_kernel, inducing_patches, num_latents=1,
                         q_mu=q_mu, q_sqrt=q_sqrt, mean_function=mean_function)

        self.base_kernel_class = base_kernel_class
        self.patch_size= patch_size

    def describe(self):
        desc = "\n\t+ Conv: patch {}".format(self.patch_size)
        desc += " base_kern {}".format(self.base_kernel_class.__name__)
        return super().describe() + desc


class IndexedConvLayer(GPLayer):

    def __init__(self,
                 input_shape: List,
                 output_shape: List,
                 number_inducing: int,
                 patch_size: List, *,
                 stride: int = 1,
                 num_filters: int = 1,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
                 # base_kernel_class: type = gpflow.kernels.RBF,
                 # index_kernel_class: type = gpflow.kernels.RBF,
                 patches_initializer: Optional[Union[np.ndarray, init.Initializer]] \
                         = init.NormalInitializer()):
        """
        This layer constructs a convolutional GP layer.
        :input_shape: tuple
            shape of the input images, W x H
        :output_shape: tuple
            shape of the output images
        :param patch_size: tuple
            Shape of the patches (a.k.a kernel_size of filter_size)
        :param number_inducing: int
            Number of inducing patches, M

        Optional:
        :param stride: int
            An integer specifying the strides of the convolution along the height and width.
        :param num_filters: int
            Number of filters in the convolution
        :param q_mu and q_sqrt: np.ndarrays
            Variatial posterior parameterisation.
        :param patches_initializer: init.Initializer or np.ndarray
            Instance of the class `init.Initializer` that initializes the inducing patches,
            can also be a np.ndarray, if this is the case the patches_initializer param
            holds the inducing patches M x w x h
        """
        # assert num_filters == 1 and stride == 1  # TODO
        assert stride == 1  # TODO
        assert len(output_shape) == 2, "Index kernel defined over 2-dim indices"
        assert output_shape[0] == output_shape[1], "Square images are supported only"

        if not _check_input_output_shape(input_shape, output_shape, patch_size):
            print("input_shape:", input_shape)
            print("output_shape:", output_shape)
            print("patch_size:", patch_size)
            raise ValueError("The input, output and patch size are inconsistent in the ConvLayer. "
                             "The correct dimension should be: output = input - patch_size + 1.")

        # Construct feature
        Z_indices = np.random.randint(0,
                                      output_shape[0],
                                      size=(number_inducing, len(output_shape)))
        inducing_indices = gpflow.features.InducingPoints(Z_indices)
        shape = [number_inducing, *patch_size]  # tuple with values: M x w x h
        init_patches = _from_patches_initializer_to_patches(patches_initializer, shape)  # M x w x h
        inducing_patches = InducingPatch(init_patches)
        feature = IndexedInducingPatch(inducing_patches, inducing_indices)

        base_kernel_class = gpflow.kernels.RBF
        index_kernel_class = gpflow.kernels.RBF
        # Construct kernel
        index_kernel = index_kernel_class(len(output_shape), lengthscales=10.)
        base_kernel = base_kernel_class(np.prod(patch_size))  # TODO: we are using the default kernel hyps
        conv_kernel = ConvKernel(base_kernel, input_shape, patch_size, colour_channels=1)  # TODO add colour
        kernel = IndexedConvKernel(conv_kernel, index_kernel)

        super().__init__(kernel, feature, num_latents=num_filters,
                         q_mu=q_mu, q_sqrt=q_sqrt, mean_function=mean_function)

        # Save info for future self.describe() calls
        # self.base_kernel_class = base_kernel_class
        self.patch_size= patch_size
        # self.index_kernel_class = index_kernel_class

    def describe(self):
        desc = "\n\t+ Indexed Conv: patch {}".format(self.patch_size)
        # desc += " base_kern {}".format(self.base_kernel_class.__name__)
        # desc += "\n\t+ index_kern: {}".format(self.index_kernel_class.__name__)
        return super().describe() + desc

    @params_as_tensors
    def propagate(self, X, *, sampling=True, full_output_cov=False, full_cov=False, **kwargs):
        """
        :param X: N x P
        """
        if sampling:
            sample = sample_conditional(X, self.feature, self.kern,
                                        self.q_mu, q_sqrt=self.q_sqrt,
                                        full_output_cov=True, white=True)
            return sample + self.mean_function(X)  # N x P
        else:
            mean, var = conditional(X, self.feature, self.kern, self.q_mu,
                                    q_sqrt=self.q_sqrt, full_cov=full_cov,
                                    full_output_cov=full_output_cov, white=True)
            return mean + self.mean_function(X), var  # N x P, variance depends on args

class PoolingIndexedConvLayer(IndexedConvLayer):

    def __init__(self,
                 input_shape: List,
                 number_inducing: int,
                 patch_size: List, *,
                 stride: int = 1,
                 num_filters: int = 1,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
                 # base_kernel_class: type = gpflow.kernels.RBF,
                 # index_kernel_class: type = gpflow.kernels.RBF,
                 patches_initializer: Optional[Union[np.ndarray, init.Initializer]] \
                         = init.NormalInitializer()):
        """
        This layer constructs a Pooling Indexed Convolutional GP layer.
        :input_shape: tuple
            shape of the input images, W x H
        :param patch_size: tuple
            Shape of the patches (a.k.a kernel_size of filter_size)
        :param number_inducing: int
            Number of inducing patches, M

        Optional:
        :param stride: int
            An integer specifying the strides of the convolution along the height and width.
        :param num_filters: int
            Number of filters in the convolution
        :param q_mu and q_sqrt: np.ndarrays
            Variatial posterior parameterisation.
        :param patches_initializer: init.Initializer or np.ndarray
            Instance of the class `init.Initializer` that initializes the inducing patches,
            can also be a np.ndarray, if this is the case the patches_initializer param
            holds the inducing patches M x w x h
        """
        output_shape = [input_shape[0] - patch_size[0] + 1,
                        input_shape[1] - patch_size[1] + 1]

        super().__init__(input_shape,
                         output_shape,
                         number_inducing,
                         patch_size,
                         stride=stride,
                         num_filters=num_filters,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         mean_function=mean_function,
                         # base_kernel_class=base_kernel_class,
                         # index_kernel_class=index_kernel_class,
                         patches_initializer=patches_initializer)

        # Use the Pooling Index kernel instead of the normal Indexed one
        self.kern = PoolingIndexedConvKernel(self.kern.conv_kernel,
                                             self.kern.index_kernel)


    def describe(self):
        desc = "\n\t+ Pooling Indexed Conv: patch {}".format(self.patch_size)
        # desc += " base_kern {}".format(self.base_kernel_class.__name__)
        # desc += "\n\t+ index_kern: {}".format(self.index_kernel_class.__name__)
        return super().describe() + desc
