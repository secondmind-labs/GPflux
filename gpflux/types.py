# Copyright (C) Secondmind Ltd 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
Types used within GPflux (for static type-checking).
"""

from typing import List, Tuple, Union

import tensorflow as tf

from gpflow.base import TensorType

ShapeType = Union[tf.TensorShape, List[int], Tuple[int, ...]]
r""" Union of valid types for describing the shape of a `tf.Tensor`\ (-like) object """

ObservationType = List[TensorType]
"""
Type for the ``[inputs, targets]`` list used by :class:`~gpflux.layers.LayerWithObservations`
"""
