import functools

import gpflow
import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

import gpflux
from refreshed_experiments.configs import ConvGPConfig, MNISTCNNConfig, CifarCNNConfig
from refreshed_experiments.data_infrastructure import ImageClassificationDataset
from refreshed_experiments.utils import reshape_to_2d, labels_onehot_to_int


@gpflow.defer_build()
def convgp_creator(dataset: ImageClassificationDataset, config: ConvGPConfig):
    num_classes = dataset.num_classes
    # DeepGP class expects 2d inputs and labels encoded with integers
    x_train, y_train = reshape_to_2d(dataset.train_features), \
                       labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
    h = int(x_train.shape[1] ** .5)
    likelihood = gpflow.likelihoods.SoftMax(num_classes)
    patches = config.patch_initializer(x_train[:100], h, h, config.init_patches)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [h, h],
        config.num_inducing_points,
        config.patch_shape,
        num_latents=num_classes,
        with_indexing=config.with_indexing,
        with_weights=config.with_weights,
        patches_initializer=patches)

    layer0.kern.basekern.variance = 25.0
    layer0.kern.basekern.lengthscales = 1.2

    if config.with_indexing:
        layer0.kern.index_kernel.variance = 25.0
        layer0.kern.index_kernel.lengthscales = 3.0

    # break symmetry in variational parameters
    layer0.q_sqrt = layer0.q_sqrt.read_value()
    layer0.q_mu = np.random.randn(*layer0.q_mu.read_value().shape)
    model = gpflux.DeepGP(x_train, y_train,
                          layers=[layer0],
                          likelihood=likelihood,
                          batch_size=config.batch_size,
                          name="my_deep_gp")
    return model


def mnist_cnn_creator(dataset: ImageClassificationDataset, config: MNISTCNNConfig):
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

    def top2_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top3_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=config.optimiser,
                  metrics=[keras.metrics.categorical_accuracy,
                           top2_accuracy,
                           top3_accuracy])
    return model


def cifar_cnn_creator(dataset: ImageClassificationDataset, config: CifarCNNConfig):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=dataset.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dataset.num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=config.optimiser,
                  metrics=['accuracy'])

    return model
