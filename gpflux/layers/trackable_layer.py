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
"""Utility layer that tracks variables in :class:`tf.Module`."""

from deprecated import deprecated

from gpflow.keras import tf_keras


@deprecated(
    reason=(
        "GPflux's `TrackableLayer` was prior to TF2.5 used to collect GPflow "
        "variables in subclassed layers. As of TF 2.5, `tf.Module` supports "
        "this natively and there is no need for `TrackableLayer` anymore. It will "
        "be removed in GPflux version `1.0.0`."
    )
)
class TrackableLayer(tf_keras.layers.Layer):
    """
    With the release of TensorFlow 2.5, our TrackableLayer workaround is no
    longer needed.  See https://github.com/Prowler-io/gpflux/issues/189.
    Will be removed in GPflux version 1.0.0
    """

    pass
