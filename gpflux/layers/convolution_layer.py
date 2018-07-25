import numpy as np
import gpflow

from .. import init
from ..convolution_kernel import ConvKernel
from ..inducing_patch import InducingPatch

from .layers import GPLayer


def _check_input_output_shape(input_shape, output_shape, patch_size):
    print('input_shape' , input_shape)
    print('patch_size ', patch_size)

    width_check = (input_shape[0] - patch_size[0] + 1 == output_shape[0])
    height_check = (input_shape[1] - patch_size[1] + 1 == output_shape[1])
    return width_check and height_check

class ConvLayer(GPLayer):

    def __init__(self,
                 input_shape,
                 output_shape,
                 number_inducing,
                 patch_size, *,
                 stride=1,
                 num_filters=1,
                 q_mu=None,
                 q_sqrt=None,
                 mean_function=None,
                 base_kernel_class=gpflow.kernels.RBF,
                 inducing_patches_initializer=init.NormalInitializer()):
        """
        This layer constructs a convolutional GP layer.
        :input_shape: tuple
            shape of the input images, W x H
        :param patch_size: tuple
            Shape of the patches (a.k.a kernel_size of filter_size)
        :param number_inducing: int
            Number of inducing patches, M

        Optional:
        :param stride: int
            An integer specifying the strides of the convolution along the height and width.
            TODO: not implemeted yet
        :param num_filters: int
            Number of filters in the convolution
        :param q_mu and q_sqrt: np.ndarrays
            Variatial posterior parameterisation.
        :param inducing_patches_initializer: init.Initializer.
            Instance of the class `init.Initializer` that initializes the inducing patches.
        """
        assert num_filters == 1 and stride == 1  # TODO

        if not _check_input_output_shape(input_shape, output_shape, patch_size):
            print("input_shape: ", input_shape)
            print("output_shape: ", output_shape)
            print("patch_size: ", patch_size)
            raise ValueError("The input, output and patch size are inconsistent in the ConvLayer. "
                             "The correct dimension should be: output = input - patch_size + 1.")

        # inducing patches
        inducing_patch_shape = [number_inducing, *patch_size]  # tuple with values: M x w x h
        init_patches = inducing_patches_initializer(inducing_patch_shape)  # M x w x h
        inducing_patches = InducingPatch(init_patches)

        base_kernel = base_kernel_class(np.prod(patch_size))  # TODO: we are using the default kernel hyps
        conv_kernel = ConvKernel(base_kernel, input_shape, patch_size, colour_channels=1)  # TODO add colour

        super().__init__(conv_kernel, inducing_patches, num_latents=1,
                         q_mu=q_mu, q_sqrt=q_sqrt, mean_function=mean_function)

        self.base_kernel_class = base_kernel_class
        self.patch_size= patch_size

    def describe(self):
        s = super().describe()
        s += "\n\t+ Conv: patch {}, base_kern {}".\
                format(self.patch_size, self.base_kernel_class.__name__)
        return s

