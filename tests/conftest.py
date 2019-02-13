# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import pytest
import tensorflow as tf


@pytest.fixture
def session_tf():
    gpu_opts = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_opts)
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph, config=config).as_default() as session:
            yield session
