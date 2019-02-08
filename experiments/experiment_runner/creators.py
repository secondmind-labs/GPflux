# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import abc
import gpflow
from gpflow.kullback_leiblers import gauss_kl
from gpflow import params_as_tensors
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import gpflux
from experiments.experiment_runner.utils import reshape_to_2d, labels_onehot_to_int
from gpflux.layers import ConvLayer, WeightedSumConvLayer

import tensorflow as tf


class ModelCreator(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def create(data_source, config):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class BasicCNN(ModelCreator):
    @staticmethod
    def create(data_source, config):
        dataset = data_source.get_data()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=dataset.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(dataset.num_classes, activation='softmax'))
        return model



def create_convgp_layer(input_size, num_ind_points, patch_shape, num_classes, with_indexing,
                        with_weights,
                        patches):
    layer0 = gpflux.layers.WeightedSumConvLayer(
        input_size,
        num_ind_points,
        patch_shape,
        num_latents=num_classes,
        with_indexing=with_indexing,
        with_weights=with_weights,
        patches_initializer=patches)
    layer0.kern.basekern.variance = 25.0
    layer0.kern.basekern.lengthscales = 1.2

    if with_indexing:
        layer0.kern.index_kernel.variance = 25.0
        layer0.kern.index_kernel.lengthscales = 3.0

    # break symmetry in variational parameters
    layer0.q_sqrt = layer0.q_sqrt.read_value()
    layer0.q_mu = np.random.randn(*layer0.q_mu.read_value().shape)
    return layer0


class ShallowConvGP(ModelCreator):
    @staticmethod
    @gpflow.defer_build()
    def create(data_source, config):
        dataset = data_source.get_data()
        num_classes = dataset.num_classes
        # DeepGP class expects 2d inputs and labels encoded with integers
        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
        h = int(x_train.shape[1] ** .5)
        likelihood = gpflow.likelihoods.SoftMax(num_classes)
        patches = config.patch_initializer(x_train[:100], h, h, config.init_patches)
        layer0 = create_convgp_layer(
            [h, h],
            config.num_inducing_points,
            config.patch_shape,
            num_classes,
            config.with_indexing,
            config.with_weights,
            patches)

        model = gpflux.DeepGP(x_train, y_train,
                              layers=[layer0],
                              likelihood=likelihood,
                              batch_size=config.batch_size,
                              name="conv_gp")
        return model
