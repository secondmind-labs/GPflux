# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import time
import os
from typing import Tuple, Callable, Dict, Optional, List

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras

import gpflow
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflux.initializers import NormalInitializer
from gpflux.layers.convolution_layer import WeightedSumConvLayer
from gpflux.models.deep_gp import DeepGP
from gpflux.layers import GPLayer

SEED = 0  # used seed to ensure that there's no variance in timing coming from randomness


def _get_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility method to obtain MNIST data.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / x_train.max()
    y_train = np.diag(np.ones(10))[y_train]
    return x_train, y_train, x_test, y_test


def _get_timing_for_fixed_op(num_optimisation_updates: int, session: tf.Session, op: tf.Operation) \
        -> Callable[[], float]:
    """
    Returns a method that runs op using session for num_optimisation_updates. The returned
    method will return its execution time.
    """

    def profile():
        t0 = time.time()
        for _ in range(num_optimisation_updates):
            session.run(op)
        t1 = time.time()
        return t1 - t0

    return profile


def _get_convgp_profile_method(with_indexing: bool, num_optimisation_updates: int = 20) \
        -> Callable[[], float]:
    """
    Returns a method that performs num_optimisation_updates on Conv GP.
    """
    gpflow.reset_default_graph_and_session()
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    batch_size = 32
    with_weights = True
    x, y, *_ = _get_mnist()
    num_inducing_points = 200
    patch_shape = [5, 5]
    likelihood = gpflow.likelihoods.SoftMax(y.shape[1])

    patches = NormalInitializer()
    layer = WeightedSumConvLayer(
        x.shape[1:3],
        num_inducing_points,
        patch_shape,
        num_latent=likelihood.num_classes,
        with_indexing=with_indexing,
        with_weights=with_weights,
        patches_initializer=patches)

    layer.q_sqrt = layer.q_sqrt.read_value()
    layer.q_mu = np.random.randn(*layer.q_mu.read_value().shape)

    x = x.reshape(x.shape[0], -1)  # DeepGP class expects two dimensional data

    model = DeepGP(x, y,
                   layers=[layer],
                   likelihood=Gaussian(),
                   minibatch_size=batch_size,
                   name="my_deep_gp")
    model.compile()
    optimizer = gpflow.train.AdamOptimizer()
    op = optimizer.make_optimize_tensor(model)
    session = model.enquire_session()
    return _get_timing_for_fixed_op(num_optimisation_updates, session, op)


def _get_svgp_rbf_profile_method(num_optimisation_updates: int = 20) -> Callable[[], float]:
    """
    Returns a method that performs num_optimisation_updates on SVGP.
    """
    gpflow.reset_default_graph_and_session()
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    batch_size = 32
    x, y = np.random.random((1000, 10)), np.random.random((1000, 10))
    inducing_feature = InducingPoints(np.random.random((5, 10)))
    kernel = RBF(input_dim=x.shape[1])
    layer = GPLayer(kernel, feature=inducing_feature, num_latent=y.shape[1])
    model = DeepGP(x, y,
                   layers=[layer],
                   likelihood=Gaussian(),
                   minibatch_size=batch_size,
                   name="my_deep_gp")
    model.compile()
    optimizer = gpflow.train.AdamOptimizer()
    op = optimizer.make_optimize_tensor(model)
    session = model.enquire_session()
    return _get_timing_for_fixed_op(num_optimisation_updates, session, op)


class TimingTask:
    """
    Class representing a task to be timed.
    """

    def __init__(self, name: str, creator: Callable, num_iterations: int,
                 num_warm_up: int, creator_args: Optional[Dict] = None) -> None:
        """
        Create a timed task.
        :param name: the name of the task
        :param creator: this callable should return a method to be timed; this method should return
        its execution time
        :param num_iterations: the number of repetitions to calculate average running time
        :param num_warm_up: the number of initial iterations that will be ignored
        :param creator_args: dictionary of additional arguments passed to creator as **kwargs
        """
        self.num_iterations = num_iterations
        self.num_warm_up = num_warm_up
        self.name = name
        self.creator = creator
        self.creator_args = {} if creator_args is None else creator_args
        assert self.num_iterations > self.num_warm_up, \
            'Number of iterations has to be greater than the number of warm up repetitions'


class Timer:
    """
    Timer runs a passed list of timing tasks.
    """

    def __init__(self, task_list) -> None:
        """
        Initialize the Timer by passing a list of tasks to time.
        """
        self._task_list = task_list

    def run(self) -> str:
        """
        Runs tasks on a task_list and returns the report string containing execution times of tasks.
        """
        report_str = 'Timings:'
        for task in self._task_list:
            times = []
            for i in tqdm(range(task.num_iterations), desc='Running task {}'.format(task.name),
                          ncols=80):
                profiled_method = task.creator(**task.creator_args)
                t = profiled_method()
                if i < task.num_warm_up:
                    continue
                times.append(t)
            times = [t * 1000 for t in times]  # convert to ms
            report_str += '\nTask for {}: mean {:.3f} ms, std {:.3f} ms'.format(task.name,
                                                                                np.mean(times),
                                                                                np.std(times))
        return report_str


def get_timing_tasks(num_optimisation_updates: int, num_iterations: int, num_warm_up: int) \
        -> List[TimingTask]:
    """
    This method defines tasks that we want to profile.
    """
    timing_tasks = \
        [
            TimingTask(name='profile SVGP RBF',
                       creator=_get_svgp_rbf_profile_method,
                       num_iterations=num_iterations,
                       num_warm_up=num_warm_up),
            TimingTask(name='profile CONV GP',
                       creator=_get_convgp_profile_method,
                       creator_args=dict(with_indexing=False,
                                         num_optimisation_updates=num_optimisation_updates),
                       num_iterations=num_iterations,
                       num_warm_up=num_warm_up),
            TimingTask(name='profile CONV GP TICK',
                       creator=_get_convgp_profile_method,
                       creator_args=dict(with_indexing=True,
                                         num_optimisation_updates=num_optimisation_updates),
                       num_iterations=num_iterations,
                       num_warm_up=num_warm_up)
        ]
    return timing_tasks


def _run_timings() -> None:
    """
    Entrypoint for profiling. This is run by Makefile.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    timing_tasks = get_timing_tasks(num_optimisation_updates=20,
                                    num_iterations=30,
                                    num_warm_up=10)
    timer = Timer(task_list=timing_tasks)
    print(timer.run())


if __name__ == '__main__':
    _run_timings()
