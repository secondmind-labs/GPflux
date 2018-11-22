import datetime

import gpflow
import keras
from gpflow.training import monitor as mon
import gpflux


class _Configuration:

    def summary(self):
        summary_str = ['Configuration parameters:\n']
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            if hasattr(value, '__call__'):
                summary_str.append('{} {}\n'.format(name, value.__name__))
            else:
                summary_str.append('{} {}\n'.format(name, str(value)))
        return ''.join(summary_str)

    @property
    def name(self):
        return self.__class__.__name__


class GPConfig(_Configuration):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 1
        self.num_inducing_points = 100
        self.monitor_stats_num = 1000
        self.lr = 0.01
        self.store_frequency = 2000

    def get_optimiser(self, step):
        lr = self.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)

    def get_tasks(self, x, y, model, path):
        return []


class RBFGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.num_epochs = 550
        self.num_inducing_points = 1000
        self.lr = 0.01

    def get_optimiser(self, step):
        lr = self.lr * 1.0 / (1 + step // 5000 / 3)
        return gpflow.train.AdamOptimizer(lr)


class ConvGPConfig(GPConfig):
    def __init__(self):
        super().__init__()
        self.batch_size = 110
        self.patch_shape = [5, 5]
        self.num_epochs = 550
        self.num_inducing_points = 1000
        self.base_kern = "RBF"
        self.with_weights = True
        self.with_indexing = False
        self.init_patches = "patches-unique"  # 'patches', 'random'

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    def get_tasks(self, x, y, model, path):
        from refreshed_experiments.utils import calc_multiclass_error
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


class NNConfig(_Configuration):
    def __init__(self):
        self.epochs = 1
        self.batch_size = 128
        self.optimiser = keras.optimizers.Adam()
        self.validation_proportion = 0.00
        self.callbacks = []
        self.early_stopping = False
        self.keras_log_dir = '/tmp/logs{}'.format(str(datetime.datetime.now()).replace(' ', '_'))
        self.callbacks = [keras.callbacks.TensorBoard(log_dir=self.keras_log_dir)]


class EarlyStoppingNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.validation_proportion = 0.1
        self.callbacks += [keras.callbacks.EarlyStopping(patience=3)]


class EarlyStoppingAccuracyNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 200
        self.validation_proportion = 0.1
        self.callbacks += [keras.callbacks.EarlyStopping(patience=3, monitor='val_acc')]


class BasicCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 12


class BasicCNNLongConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 100


class MNISTCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 12
        self.optimiser = keras.optimizers.Adadelta()


class CifarCNNConfig(NNConfig):
    def __init__(self):
        super().__init__()
        self.epochs = 200
        self.optimiser_parameters = {'lr': 0.0001, 'decay': 1e-6}
        self.optimiser = keras.optimizers.rmsprop(**self.optimiser_parameters)
