# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow
import keras
from gpflow.training import monitor as mon

import gpflux
from experiments.experiment_runner.utils import Configuration
from experiments.experiment_runner.utils import calc_multiclass_error


class KerasConfig(Configuration):

    def __init__(self):
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.callbacks = []
        self.epochs = 12


class GPConfig(Configuration):

    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 1
        self.num_inducing_points = 100
        self.monitor_stats_num = 1000
        self.lr = 0.01
        self.store_frequency = 2000
        self.optimiser = gpflow.train.AdamOptimizer

    def get_learning_rate(self, step):
        return self.lr * 1.0 / (1 + step // 5000 / 3)


class ConvGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 110
        self.patch_shape = [5, 5]
        self.num_epochs = 3
        self.num_inducing_points = 1000
        self.base_kern = "RBF"
        self.with_weights = True
        self.with_indexing = False
        self.init_patches = "patches-unique"  # 'patches-unique', 'random'

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    def get_tasks(self, x, y, model, path):
        stats_fraction = self.monitor_stats_num
        tasks = []
        fw = mon.LogdirWriter(str(path))

        tasks += [
            mon.CheckpointTask(str(path))
                .with_name('saver')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))]

        tasks += [
            mon.ModelToTensorBoardTask(fw, model)
                .with_name('model_tboard')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))
                .with_exit_condition(True)
                .with_flush_immediately(True)]

        def error_func(*args, **kwargs):
            xs, ys = x[:stats_fraction], y[:stats_fraction]
            return calc_multiclass_error(model, xs, ys, batchsize=50)

        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, error_func, "error")
                .with_name('error')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))
                .with_exit_condition(True)
                .with_flush_immediately(True)]
        return tasks


class TickConvGPConfig(ConvGPConfig):
    def __init__(self):
        super().__init__()
        self.with_indexing = True


class RBFGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 4
        self.num_inducing_points = 1000
        self.lr = 0.01
