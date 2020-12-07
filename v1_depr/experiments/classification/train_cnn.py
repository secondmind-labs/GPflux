# pylint: disable=E1120,W0612

import datetime
import math
import os
from pathlib import Path

import cnn
import datasets
import numpy as np
import tensorflow.keras.backend as K
from sacred import Experiment
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

NAME = "cnn-training"
ex = Experiment(NAME)


@ex.config
def config():
    basepath = './cnns'
    date = datetime.datetime.now().strftime('%d-%H-%M')
    dataset = 'cifar10'
    batch_size = 128
    epochs = 600
    steps_per_epoch = 10000
    optimizer = "adam"  # adam, sgd
    lr = dict(
        mode = "step_decay", # custom, step_decay, constant
        lr = 0.003,
        decay_drop = 0.9,
        epochs_decay_drop = 10.
    )

    augment = dict(
        factor = 1,
        use_shift = False,
        use_shear = False,
        use_zoom = False,
        use_rotation = False)


@ex.capture
def path(basepath, date, optimizer, lr, augment):
    names = [
        ('opt', optimizer),
        ('lr', lr['lr']),
        ('lrmode', lr['mode']),
    ]

    if lr['mode'] == "step_decay":
        names.append(('lrdrop', lr['decay_drop']))
        names.append(('lredrop', lr['epochs_decay_drop']))

    if do_augmentation():
        for k, v in augment.items():
            if not k.startswith('use_'):
                names.append((k, v))
            elif v:
                names.append((k.split('_')[-1], 'on'))

    path = str(Path(basepath, date))

    def experiment_path(basepath: str, name: str, kwargs) -> str:
        assert isinstance(name, str)
        for (k, v) in kwargs:
            name += "-{0}_{1}".format(k, v)
        return str(Path(basepath, name))

    return experiment_path(path, "cnn", names)


@ex.capture
def get_data(dataset):
    return datasets.get_dataset(dataset, x_type=np.float32)


@ex.capture
def cnn_experiment_name(lr, dataset, batch_size, epochs):
    dict_args = {'lr': np.array(lr['lr']), 'batch_size': batch_size, 'epochs': epochs}
    dict_args = ['{}={}'.format(k, v) for k, v in dict_args.items()]
    return '_'.join([dataset, 'cnn', *dict_args])


@ex.capture
def step_decay(lr):
    def callback(epoch):
        return lr['lr'] * math.pow(lr['decay_drop'], math.floor((1 + epoch) / lr['epochs_decay_drop']))
    return callback


class TensorBoardWithLR(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


@ex.capture
def cnn_monitor_callbacks(dataset, lr):
    p = path()
    filename = '{0}/cnn.{1}.{2}'.format(p, dataset, '.{epoch:02d}-{acc:.2f}.h5')
    cbs = []
    logdir = os.path.join(p, 'tb')
    cbs.append(ModelCheckpoint(filename, verbose=1, period=50))
    cbs.append(TensorBoardWithLR(logdir))
    if lr['mode'] == "step_decay":
        cbs.append(LearningRateScheduler(step_decay()))
    else:
        cbs.append(LearningRateScheduler(lambda _: lr['lr']))
    return cbs


@ex.capture
def do_augmentation(augment):
    return any([v for k, v in augment.items() if k.startswith('use_')])


@ex.capture
def make_augment_cb(augment):
    def augment_cb(x):
        return datasets.augment_image(x,
                augment_factor=augment['factor'],
                use_shear=augment['use_shear'],
                use_rotation=augment['use_rotation'],
                use_shift=augment['use_shift'],
                use_zoom=augment['use_zoom'])
    return augment_cb


@ex.capture
def fit_cnn_model(nn, train, test, batch_size, steps_per_epoch, epochs, optimizer):
    x, y = train
    callbacks = cnn_monitor_callbacks()
    nn.compile(optimizer=optimizer,
               loss='sparse_categorical_crossentropy',
               metrics=['mse', 'accuracy'])

    if do_augmentation():
        augment_cb = make_augment_cb()
        gen = datasets.ImageGenerator(x, y, batch_size=batch_size, augment_cb=augment_cb)
        nn.fit_generator(gen,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test)
    else:
        nn.fit(x, y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test)


    xt, yt = test
    test_metrics = nn.evaluate(xt, yt, batch_size=batch_size)
    print(">>> Test metrics: ", test_metrics)

@ex.automain
def main():
    train_data, test_data = get_data()
    nn = cnn.cifar10_cnn_model()
    fit_cnn_model(nn, train_data, test_data)
    return 0