import math
import os
from typing import Tuple

import numpy as np
import observations as obs
import tensorflow as tf
import tensorflow.keras.backend as K
from sacred import Experiment
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (Callback, LearningRateScheduler,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.initializers import constant, glorot_normal
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPool2D)
from tensorflow.keras.regularizers import l2

DatasetTrain = Tuple[np.ndarray, np.ndarray]
DatasetTest = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[DatasetTrain, DatasetTest]


NAME = "cifar"
ex = Experiment(NAME)


@ex.config
def config():
    basepath = '.'
    dataset = 'cifar10'
    batch_size = 128
    epochs = 100
    steps_per_epoch = 10000
    lr_mode = "custom"
    lr = 0.003
    step_width = 50000
    step_scale = 3.
    hz = 50
    hz_slow = 500
    restore = False
    step_decay_drop=0.5
    step_epochs_decay_drop=10.


@ex.capture
def data(basepath, dataset) -> Dataset:
    tmp_dir = os.path.join('/tmp', dataset)
    if dataset == 'cifar10':
        (x, y), (xt, yt) = obs.cifar10(tmp_dir)
        ny = 10
    axes = (0, 2, 3, 1)
    b = np.zeros((len(y), 4))

    def upd_x(d): return np.transpose(d, axes).astype(np.float32) / 255.
    x, xt = upd_x(x), upd_x(xt)

    def upd_y(d):
        cd = d.reshape((-1, 1))
        return cd
    y, yt = upd_y(y), upd_y(yt)
    return (x, y), (xt, yt)


@ex.capture
def cnn_experiment_name(lr, dataset, batch_size, epochs):
    dict_args = {'lr': np.array(lr), 'batch_size': batch_size, 'epochs': epochs}
    dict_args = ['{}={}'.format(k, v) for k, v in dict_args.items()]
    return '_'.join([dataset, 'cnn', *dict_args])


@ex.capture
def step_decay(lr, step_decay_drop, step_epochs_decay_drop):
    def callback(epoch):
        return lr * math.pow(step_decay_drop, math.floor((1 + epoch) / step_epochs_decay_drop))
    return callback



class CustomLearningRateScheduler(Callback):
    def __init__(self, logdir='.', lr=1e-2, step_width=10000, step_scale=3., each_iter=1000):
        super().__init__()
        self.lr = lr
        self.lr_writer = tf.summary.FileWriter(logdir, filename_suffix="lr")
        self.lr_logdir = logdir
        self.step_width = step_width
        self.step_scale = step_scale
        self.iteration = 0
        self.each_iter = each_iter

    def clr(self):
        if self.step_scale != 0 and self.step_width != 0:
            return self.lr * 1.0 / (1 + self.iteration // self.step_width / self.step_scale)
        return self.lr

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        self.iteration += 1
        lr = self.clr()
        K.set_value(self.model.optimizer.lr, lr)

        if self.iteration % self.each_iter == 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=lr)])
            self.lr_writer.add_summary(summary, self.iteration)
            self.lr_writer.flush()
            print("\n >>> Learning rate: {}".format(lr))

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.lr_writer.close()


@ex.capture
def monitor_callbacks(basepath, dataset, lr_mode, lr, step_width, step_scale):
     = experiment_path()
    filename = 'cnn.' + dataset + '.{epoch:02d}-{acc:.2f}.h5'

    cbs = []
    logdir = './tensorboard/'
    cbs.append(ModelCheckpoint(filename, verbose=1))
    cbs.append(TensorBoard(logdir))
    if lr_mode == "custom":
        cbs.append(CustomLearningRateScheduler(logdir=logdir, lr=lr, step_width=step_width, step_scale=step_scale))
    elif lr_mode == "step_decay":
        cbs.append(LearningRateScheduler(step_decay()))
    else:
        cbs.append(LearningRateScheduler(lambda _: lr))
    return cbs



@ex.capture
def fit_model(nn, train, test, batch_size, steps_per_epoch, epochs):
    x, y = train
    callbacks = monitor_callbacks()
    nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['mse', 'accuracy'])
    nn.fit(x, y,
           batch_size=batch_size,
           epochs=epochs,
           callbacks=callbacks,
           validation_data=test)


def cifar10_cnn_model():
    nn = Sequential()
    nn.add(Conv2D(64, (3, 3), (1, 1), activation='relu',
           padding='same', input_shape=(32, 32, 3), kernel_initializer=glorot_normal()))
    nn.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    nn.add(BatchNormalization())

    nn.add(Conv2D(64, (5, 5), (1, 1), activation='relu',
           padding='same', kernel_initializer=glorot_normal()))
    nn.add(BatchNormalization())
    nn.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    nn.add(Flatten())
    nn.add(Dense(384, kernel_regularizer=l2(0.004),
           bias_initializer=constant(0.1), kernel_initializer=glorot_normal()))
    nn.add(Dense(192, kernel_regularizer=l2(0.004),
           bias_initializer=constant(0.1), kernel_initializer=glorot_normal()))
    nn.add(Dropout(0.25))
    nn.add(Dense(10, activation='softmax', kernel_initializer=glorot_normal()))
    return nn


@ex.automain
def main():
    train_data, test_data = data()
    nn = cifar10_cnn_model()
    fit_model(nn, train_data, test_data)
    return 0
