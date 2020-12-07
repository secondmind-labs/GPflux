import numpy as np
import pytest
import tensorflow as tf

from . import util


@pytest.mark.parametrize('op_type', ['conv', 'naive'])
def test_image_norm_benchmark(op_type, benchmark):
    image_shape = (32, 32, 1)
    patch_shape = (5, 5)

    n = 64
    x = np.random.randn(n, *image_shape)

    gpu_opts = tf.GPUOptions(allow_growth=True) # , per_process_gpu_memory_fraction=0.333)
    tf_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_opts)
    with tf.Session(graph=tf.Graph(), config=tf_config) as session:
        tx = tf.convert_to_tensor(x)
        op = util.create_op(op_type, image_shape, patch_shape, tx)
        def run_norm():
            session.run(op)
        benchmark.pedantic(run_norm, warmup_rounds=5, rounds=100, iterations=10)
