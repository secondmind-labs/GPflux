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
TensorBoard event iterator
"""
from dataclasses import dataclass
from typing import Any, Iterator, List, Type, Union

import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util


@dataclass
class Event:
    """Minimal container to hold TensorBoard event data"""

    tag: str  # summary name, e.g. "loss" or "lengthscales"
    step: int
    value: Any
    dtype: Type


def tensorboard_event_iterator(file_pattern: Union[str, List[str], tf.Tensor]) -> Iterator[Event]:
    """
    Iterator yielding preprocessed tensorboard Events.

    :param file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
    """

    def get_scalar_value(value: Any) -> Any:
        # Note(Vincent): I'm sorry this is messy...
        # Using `value.simple_value` returns 0.0 for
        # np.ndarray values, so we need to try `MakeNdarray`
        # first, which breaks for non-tensors.
        try:
            v = tensor_util.MakeNdarray(value.tensor).item()
        except Exception:
            try:
                v = value.simple_value
            except Exception:
                raise ValueError("Unable to read value from tensor")
        return v

    event_files = tf.data.Dataset.list_files(file_pattern)
    serialized_examples = tf.data.TFRecordDataset(event_files)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            v = get_scalar_value(value)
            yield Event(tag=value.tag, step=event.step, value=v, dtype=type(v))
