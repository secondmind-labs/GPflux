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

    def __init__(self, batch_size=64, optimiser=keras.optimizers.Adam,
                 callbacks=(), num_epochs=10, steps_per_epoch=500):
        self.batch_size = batch_size
        self.optimiser = optimiser
        self.callbacks = callbacks
        self.epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch


class GPConfig(Configuration):

    def __init__(self, batch_size=110,
                 monitor_stats_num=2000,
                 num_inducing_points=1000,
                 num_epochs=500,
                 steps_per_epoch=1000,
                 store_frequency=5000,
                 lr=0.01,
                 optimiser=gpflow.train.AdamOptimizer):
        super().__init__()
        self.batch_size = batch_size
        self.monitor_stats_num = monitor_stats_num
        self.num_epochs = num_epochs
        self.num_inducing_points = num_inducing_points
        self.steps_per_epoch = steps_per_epoch
        self.store_frequency = store_frequency
        self.lr = lr
        self.optimiser = optimiser

    def get_learning_rate(self, step):
        return self.lr * 1.0 / (1 + step // 5000 / 3)


class ConvGPConfig(GPConfig):
    def __init__(self, batch_size=110,
                 patch_shape=[5, 5],
                 monitor_stats_num=1000,
                 num_inducing_points=1000,
                 num_epochs=500,
                 steps_per_epoch=1000,
                 base_kernel='RBF',
                 with_weights=True,
                 with_indexing=False,
                 init_patches='patches-unique',
                 store_frequency=5000):
        super().__init__(batch_size, monitor_stats_num, num_inducing_points, num_epochs,
                         steps_per_epoch, store_frequency)
        self.patch_shape = patch_shape
        self.base_kern = base_kernel
        self.with_weights = with_weights
        self.with_indexing = with_indexing
        self.init_patches = init_patches
        assert self.init_patches in ['patches-unique', 'patches-unique', 'random']

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    def get_tasks(self, x, y, model, path, optimizer):
        stats_fraction = self.monitor_stats_num
        tasks = []
        fw = mon.LogdirWriter(str(path))

        def lr(*args, **kwargs):
            sess = model.enquire_session()
            return sess.run(optimizer._optimizer._lr)

        def reconstruction(*args, **kwargs):
            sess = model.enquire_session()
            return sess.run(model.E_log_prob)

        def kl(*args, **kwargs):
            sess = model.enquire_session()
            return sess.run(model.KL_U_layers)

        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, lr, "lr")
                .with_name('lr')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))
                .with_exit_condition(True)
                .with_flush_immediately(True)]
        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, reconstruction, "E_log_prob")
                .with_name('E_log_prob')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))
                .with_exit_condition(True)
                .with_flush_immediately(True)]

        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, kl, "KL")
                .with_name('KL')
                .with_condition(mon.PeriodicIterationCondition(self.store_frequency))
                .with_exit_condition(True)
                .with_flush_immediately(True)]

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
    def __init__(self, with_indexing=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_indexing = with_indexing
