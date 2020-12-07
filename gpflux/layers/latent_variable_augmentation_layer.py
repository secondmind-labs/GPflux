# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
from typing import Optional, Tuple

import tensorflow as tf

from gpflow.base import TensorType

from gpflux.layers.latent_variable_layer import LatentVariableLayer


class LatentVariableAugmentationLayer(LatentVariableLayer):
    """
    A layer to augment an output with a latent dimensions.

    LatentVariableLayers are often used to draw a latent variable, and concatenate it
    to the input. This wraps a LatentVariableLayer, and puts the concatenation
    of input and sampled latent_variable in the simple Layer framework.
    """

    def call(
        self,
        data: Tuple[TensorType, TensorType],
        training: bool = False,
        seed: Optional[int] = None,
    ) -> TensorType:
        """
        When training: Draw a sample of the latent variable from the posterior,
        whose distribution is parameterized by the encoder mapping from the data
        (comprised of inputs and targets). Concatenate this sample with input data, and
        return it. Add a KL divergence to the losses.

        When not training: Draw a sample of the latent variable from the prior.

        :param data: tuple of (inputs, targets) if training else inputs
        :param training: training mode indicator

        :return: data inputs concatenated with latent variable # [..., D+W]
        """
        inputs, targets = data
        if training:
            recognition_data = tf.concat([inputs, targets], axis=-1)
        else:
            recognition_data = inputs
        latent_data = super().call(recognition_data, training=training, seed=seed)
        augmented_inputs = tf.concat([inputs, latent_data], axis=-1)
        return augmented_inputs
