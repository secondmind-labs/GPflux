# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidentialimport numpy as np

from __future__ import absolute_import

from . import utils
from . import init

from . import convolution
from . import layers

from .models.encoders import Encoder, GPflowEncoder

from .models.doubly_stochastic_deep_gp import DeepGP
