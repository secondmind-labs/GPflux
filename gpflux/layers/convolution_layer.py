# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import List, Optional, Union

import gpflow
import numpy as np

from .layers import GPLayer
from .. import init
from ..convolution import ConvKernel, WeightedSum_ConvKernel
from ..convolution import InducingPatch, IndexedInducingPatch


def _correct_input_output_shape(input_shape, output_shape, patch_size, pooling):

    if (input_shape[0] - patch_size[0] + 1) % pooling != 0:
        return False
    if (input_shape[1] - patch_size[1] + 1) % pooling != 0:
        return False
    if ((input_shape[0] - patch_size[0] + 1) / pooling) != output_shape[0]:
        return False
    if ((input_shape[1] - patch_size[1] + 1) / pooling) != output_shape[1]:
        return False

    return True



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
                 patch_size: List,
                 num_latents: int = 1,
                 *,
                 with_indexing: bool = False,
                 pooling: int = 1,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
                 base_kernel: Optional[gpflow.kernels.Kern] = None,
                 patches_initializer: Optional[Union[np.ndarray, init.Initializer]] = None):
        """
        This layer constructs a convolutional GP layer.
        :input_shape: tuple
            shape of the input images, W x H
        :output_shape: tuple
            shape of the output images
        :param patch_size: tuple
            Shape of the patches (a.k.a filter_shape of filter_size)
        :param number_inducing: int
            Number of inducing patches, M

        Optional:
        :param with_indexing: bool (default False)
            Add translation invariant kernel
        :param pooling: int (default 1: no pooling)
            Number of patches that are being summed.
            If pooling is 1 no summing of patches is happening
        :param q_mu and q_sqrt: np.ndarrays
            Variatial posterior parameterisation.
        :param patches_initializer: init.Initializer or np.ndarray (default: NormalInitializer)
            Instance of the class `init.Initializer` that initializes the inducing patches,
            can also be a np.ndarray, if this is the case the patches_initializer param
            holds the inducing patches M x w x h
        """

        if not _correct_input_output_shape(input_shape, output_shape, patch_size, pooling):
            raise ValueError("The input, output and patch size are inconsistent in the ConvLayer. "
                             "The correct dimension should be: "
                             "output = (input - patch_size + 1.) / pooling\n"
                             "input_shape: {}\noutput_shape: {}\npatch_size: {}"
                             .format(input_shape, output_shape, patch_size))
        # inducing patches
        if patches_initializer is None:
            patches_initializer = init.NormalInitializer()
        shape = [number_inducing, *patch_size]  # tuple with values: M x w x h
        patches = _from_patches_initializer_to_patches(patches_initializer, shape)  # M x w x h

        if with_indexing:
            val = input_shape[0] - patch_size[0] + 1
            indices = np.random.randint(0, val, size=[number_inducing, len(input_shape)])
            indices = indices.astype(np.float64)
            feat = IndexedInducingPatch(patches, indices)
        else:
            feat = InducingPatch(patches)

        # Convolutional kernel
        if base_kernel is None:
            base_kernel = gpflow.kernels.RBF(np.prod(patch_size))
        else:
            assert base_kernel.input_dim == np.prod(patch_size)

        kern = ConvKernel(base_kernel,
                          img_size=input_shape,
                          patch_size=patch_size,
                          pooling=pooling,
                          with_indexing=with_indexing)

        super().__init__(kern, feat, num_latents=num_latents,
                         q_mu=q_mu, q_sqrt=q_sqrt, mean_function=mean_function)

        self.with_indexing = with_indexing
        self.pooling = pooling
        self.base_kernel_type = base_kernel.__class__.__name__
        self.patch_size= patch_size

    def describe(self):
        desc = "\n\t+ Conv: patch {}".format(self.patch_size)
        desc += " base_kern {}".format(self.base_kernel_type)
        desc += " pooling {}".format(self.pooling)
        if self.with_indexing:
            desc += " with_indexing "

        return super().describe() + desc


class WeightedSum_ConvLayer(ConvLayer):

    def __init__(self,
                 input_shape: List,
                 number_inducing: int,
                 patch_size: List,
                 num_latents: int = 1,
                 *,
                 with_indexing: bool = False,
                 with_weights: bool = False,
                 pooling: int = 1,
                 q_mu: Optional[np.ndarray] = None,
                 q_sqrt: Optional[np.ndarray] = None,
                 mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
                 base_kernel: Optional[gpflow.kernels.Kern] = None,
                 patches_initializer: Optional[Union[np.ndarray, init.Initializer]] = None):
        """
        See `ConvLayer` for docstrings.
        """
        output_shape0 = (input_shape[0] - patch_size[0] + 1) // pooling
        output_shape1 = (input_shape[1] - patch_size[1] + 1) // pooling
        output_shape = [output_shape0, output_shape1]

        super().__init__(input_shape,
                         output_shape,
                         number_inducing,
                         patch_size,
                         num_latents,
                         with_indexing=with_indexing,
                         pooling=pooling,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         mean_function=mean_function,
                         base_kernel=base_kernel,
                         patches_initializer=patches_initializer)

        if base_kernel is None:
            base_kernel = gpflow.kernels.RBF(np.prod(patch_size))
        else:
            assert base_kernel.input_dim == np.prod(patch_size)

        self.kern = WeightedSum_ConvKernel(base_kernel,
                                      img_size=input_shape,
                                      patch_size=patch_size,
                                      pooling=pooling,
                                      with_indexing=with_indexing,
                                      with_weights=with_weights)

    def describe(self):
        desc = "\nWeighted"
        return super().describe() + desc

