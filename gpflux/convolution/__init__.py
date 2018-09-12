# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from . import conditionals
from .convolution_kernel import (ImageArcCosine, Convolutional,
                                 ImageBasedKernel, ImageMatern12,
                                 ImageRBF, WeightedSumConvolutional)
from .inducing_patch import IndexedInducingPatch, InducingPatch
