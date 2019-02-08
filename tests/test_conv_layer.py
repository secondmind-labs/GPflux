# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow

from gpflux.layers import ConvLayer
import tensorflow as tf


def test_conv_layer_defer_applies_to_all_tf_nodes():
    gpflow.reset_default_graph_and_session()
    # check if defer build applies to all tensors/variables created by the ConvLayer
    graph = tf.get_default_graph()
    with gpflow.defer_build():
        ConvLayer(input_shape=[5, 5],
                  output_shape=[4, 4],
                  number_inducing=5,
                  patch_shape=[2, 2],
                  with_indexing=True)

    assert len(graph.as_graph_def().node) == 0
