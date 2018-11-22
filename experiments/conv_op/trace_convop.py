import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import json
# from .util import create_op
import util

from tensorflow.python.client import timeline


def bench_with_timeline(burn=1, iterations=5, pathname="."):
    # step = tf.train.get_or_create_global_step()
    # sess.run(tf.global_variables_initializer())

    n = 64
    image_shape = (32, 32, 1)
    patch_shape = (5, 5)

    x = np.random.randn(n, *image_shape)
    tx = tf.convert_to_tensor(x)

    gpu_opts = tf.GPUOptions(allow_growth=True) # , per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_opts))

    with tf.name_scope('conv_op'):
        op = util.create_op(None, image_shape, patch_shape, tx)

    for i in range(burn):
        sess.run(op)

    for i in range(iterations):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(op, options=run_options, run_metadata=run_metadata)

        filename = str(Path(pathname, f'output_{i}.traces'))
        save_timeline(filename, run_metadata)

    # writer = tf.summary.FileWriter(pathname, tf.get_default_graph())
    # writer.flush()
    # writer.close()


def save_timeline(filename, metadata):
    trace = timeline.Timeline(step_stats=metadata.step_stats)
    with open(filename, 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True, show_dataflow=True))


bench_with_timeline(burn=100, iterations=5)
