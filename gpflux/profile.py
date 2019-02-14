# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
import time
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras

import gpflow
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflux.init import PatchSamplerInitializer
from gpflux.layers.convolution_layer import WeightedSumConvLayer
from gpflux.models.deep_gp import DeepGP
from gpflux.layers.layers import GPLayer

SEED = 0


def get_mnist():
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train / x_train.max()
    y_train = np.diag(np.ones(10))[y_train]
    return x_train, y_train


def get_timing_for_fixed_op(num_optimisation_updates, session, op):
    def profile():
        t0 = time.time()
        for _ in range(num_optimisation_updates):
            session.run(op)
        t1 = time.time()
        return t1 - t0

    return profile


def get_conv():
    gpflow.reset_default_graph_and_session()
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    num_optimisation_updates = 20
    batch_size = 32
    x, y = get_mnist()
    with_indexing = True
    with_weights = True
    num_inducing_points = 200
    patch_shape = [5, 5]
    h = int(x.shape[1] ** .5)
    likelihood = gpflow.likelihoods.SoftMax(y.shape[1])

    patches = PatchSamplerInitializer(x[:num_inducing_points], height=h, width=h, unique=True)
    layer = WeightedSumConvLayer(
        x.shape[1:3],
        num_inducing_points,
        patch_shape,
        num_latents=likelihood.num_classes,
        with_indexing=with_indexing,
        with_weights=with_weights,
        patches_initializer=patches)

    layer.kern.basekern.variance = 25.0
    layer.kern.basekern.lengthscales = 1.2

    # init kernel
    if with_indexing:
        layer.kern.spatio_indices_kernel.variance = 25.0
        layer.kern.spatio_indices_kernel.lengthscales = 3.0

    # break symmetry in variational parameters
    layer.q_sqrt = layer.q_sqrt.read_value()
    layer.q_mu = np.random.randn(*(layer.q_mu.read_value().shape))

    x = x.reshape(x.shape[0], -1)

    model = DeepGP(x, y,
                   layers=[layer],
                   likelihood=Gaussian(),
                   batch_size=batch_size,
                   name="my_deep_gp")
    model.compile()
    optimizer = gpflow.train.AdamOptimizer()
    op = optimizer.make_optimize_tensor(model)
    session = gpflow.get_default_session()
    return get_timing_for_fixed_op(num_optimisation_updates, session, op)


def get_profile_method():
    gpflow.reset_default_graph_and_session()
    batch_size = 32
    num_optimisation_updates = 20
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    x, y = np.random.random((1000, 10)), np.random.random((1000, 10))
    inducing_feature = InducingPoints(np.random.random((5, 10)))
    kernel = RBF(input_dim=x.shape[1])
    layer = GPLayer(kernel, feature=inducing_feature, num_latents=y.shape[1])
    model = DeepGP(x, y,
                   layers=[layer],
                   likelihood=Gaussian(),
                   batch_size=batch_size,
                   name="my_deep_gp")
    model.compile()
    optimizer = gpflow.train.AdamOptimizer()
    op = optimizer.make_optimize_tensor(model)
    session = gpflow.get_default_session()
    return get_timing_for_fixed_op(num_optimisation_updates, session, op)


class TimingTask:
    def __init__(self, name, creator, iterations=30, num_warm_up=10):
        self.iterations = iterations
        self.num_warm_up = num_warm_up
        self.name = name
        self.creator = creator
        assert self.iterations > self.num_warm_up, \
            'Number of iterations has to be greater than the number of warm up repetitions'


class Timer:
    def __init__(self, task_list):
        self._task_list = task_list

    def _time(self):
        report_str = 'Timings:'
        for task in self._task_list:
            times = []
            for i in tqdm(range(task.iterations), desc='Running task {}'.format(task.name)):
                profiled_method = task.creator()
                t = profiled_method()
                if i < task.num_warm_up:
                    continue
                times.append(t)
            times = [t * 1000 for t in times]  # convert to ms
            report_str += '\nTask for {}: mean {:.3f} ms, std {:.3f} ms'.format(task.name,
                                                                                np.mean(times),
                                                                                np.std(times))
        return report_str

    def time(self, report_name=None):
        if report_name is None:
            print(self._time())
        else:
            report_str = self._time()
            with Path('./{}'.format(report_name)) as f_handle:
                f_handle.write_text(report_str)


def run_timings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # timing_task = TimingTask(name='profile SVGP RBF', creator=get_profile_method)
    timing_task = TimingTask(name='profile CONV GP', creator=get_conv)
    timer = Timer(task_list=[timing_task])
    timer.time()


run_timings()
