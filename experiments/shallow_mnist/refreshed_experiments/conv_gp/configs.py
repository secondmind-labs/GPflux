import gpflow

import gpflux
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration
import gpflow.training.monitor as mon
import numpy as np

class GPConfig(Configuration):
    batch_size = 128
    optimiser = gpflow.train.AdamOptimizer(0.001)
    iterations = 5000

    @staticmethod
    def get_monitor_tasks():
        raise NotImplementedError()

    @staticmethod
    def get_optimiser():
        raise NotImplementedError()


class ConvGPConfig(GPConfig):
    model_type = "convgp"

    lr_cfg = {
        "decay": "custom",
        "lr": 1e-4
    }

    iterations = 50000
    patch_shape = [5, 5]
    batch_size = 128
    num_inducing_points = 1000
    base_kern = "RBF"
    with_weights = True
    with_indexing = True
    init_patches = "patches-unique"  # 'patches', 'random'
    restore = False

    # print hz
    hz = {
        'slow': 1000,
        'short': 50
    }

    @staticmethod
    def patch_initializer(x, h, w, init_patches):
        if init_patches == "random":
            return gpflux.init.NormalInitializer()
        unique = init_patches == "patches-unique"
        return gpflux.init.PatchSamplerInitializer(x, width=w, height=h, unique=unique)

    @staticmethod
    def get_monitor_tasks(dataset, model, optimiser):
        Xs, Ys = dataset.train_features, dataset.train_targets
        Xs, Ys = Xs.reshape(Xs.shape[0], -1), \
               Ys.reshape(Ys.shape[0], -1).astype(np.int32).argmax(axis=-1)[
                   ..., None]
        path = 'CONVGP'
        fw = mon.LogdirWriter(path)

        tasks = []

        def lr(*args, **kwargs):
            sess = model.enquire_session()
            return sess.run(optimiser._optimizer._lr)

        def periodic_short():
            return mon.PeriodicIterationCondition(ConvGPConfig.hz['short'])

        def periodic_slow():
            return mon.PeriodicIterationCondition(ConvGPConfig.hz['slow'])

        def calc_binary_error(model, Xs, Ys, batchsize=100):
            Ns = len(Xs)
            splits = Ns // batchsize
            hits = []
            for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
                p, _ = model.predict_y(xs)
                acc = ((p > 0.5).astype('float') == ys)
                hits.append(acc)
            error = 1.0 - np.concatenate(hits, 0)
            return np.sum(error) * 100.0 / len(error)

        def calc_multiclass_error(model, Xs, Ys, batchsize=100):
            Ns = len(Xs)
            splits = Ns // batchsize
            hits = []
            for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
                p, _ = model.predict_y(xs)
                acc = p.argmax(1) == ys[:, 0]
                hits.append(acc)
            error = 1.0 - np.concatenate(hits, 0)
            return np.sum(error) * 100.0 / len(error)

        def get_error_cb(model, Xs, Ys, error_func, full=False, Ns=500):
            def error_cb(*args, **kwargs):
                if full:
                    xs, ys = Xs, Ys
                else:
                    xs, ys = Xs[:Ns], Ys[:Ns]
                return error_func(model, xs, ys, batchsize=50)

            return error_cb

        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, lr, "lr")
                .with_name('lr')
                .with_condition(periodic_short())
                .with_exit_condition(True)
                .with_flush_immediately(True)]
        #
        # tasks += [
        #     mon.CheckpointTask(path)
        #         .with_name('saver')
        #         .with_condition(periodic_short())]

        tasks += [
            mon.ModelToTensorBoardTask(fw, model)
                .with_name('model_tboard')
                .with_condition(periodic_short())
                .with_exit_condition(True)
                .with_flush_immediately(True)]

        tasks += [
            mon.PrintTimingsTask().with_name('print')
                .with_condition(periodic_short())
                .with_exit_condition(True)]

        error_func = calc_binary_error if dataset == "mnist01" \
            else calc_multiclass_error

        f1 = get_error_cb(model, Xs, Ys, error_func)
        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, f1, "error")
                .with_name('error')
                .with_condition(periodic_short())
                .with_exit_condition(True)
                .with_flush_immediately(True)]
        return tasks

    @staticmethod
    def get_optimiser(step):
        if ConvGPConfig.lr_cfg['decay'] == "custom":
            print("Custom decaying lr")
            lr = ConvGPConfig.lr_cfg['lr'] * 1.0 / (1 + step // 5000 / 3)
        else:
            lr = ConvGPConfig.lr_cfg['lr']
        return gpflow.train.AdamOptimizer(lr)
