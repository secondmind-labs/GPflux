#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
""" This module provides custom exceptions used by GPflux. """


class GPLayerIncompatibilityException(Exception):
    r"""
    This exception is raised when :class:`~gpflux.layers.GPLayer` is
    misconfigured. This can be caused by multiple reasons, but common misconfigurations are:

    - Incompatible or wrong type of :class:`~gpflow.kernels.Kernel`,
      :class:`~gpflow.inducing_variables.InducingVariables` and/or
      :class:`~gpflow.mean_functions.MeanFunction`.
    - Incompatible number of latent GPs specified.
    """


class EncoderInitializationError(Exception):
    r"""
    This exception is raised by an encoder (e.g. :class:`DirectlyParameterizedNormalDiag`)
    when parameters are not initialised correctly.
    """
