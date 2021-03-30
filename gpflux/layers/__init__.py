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
"""
Layers
"""
from gpflux.layers import basis_functions
from gpflux.layers.bayesian_dense_layer import BayesianDenseLayer
from gpflux.layers.gp_layer import GPLayer
from gpflux.layers.latent_variable_layer import LatentVariableLayer, LayerWithObservations
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.layers.trackable_layer import TrackableLayer
