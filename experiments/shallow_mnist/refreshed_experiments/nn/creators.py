import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import \
    ImageClassificationDataset
from experiments.shallow_mnist.refreshed_experiments.nn.configs import MNISTCNNConfiguration, \
    CifarCNNConfiguration

"""
One would implement a model creator and corresponding config to try a new model. The rest should
be done automatically.
"""


def mnist_cnn_creator(dataset: ImageClassificationDataset, config: MNISTCNNConfiguration):
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

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=config.optimiser,
                  metrics=['accuracy'])
    return model


def mnist_fashion_cnn_creator(dataset: ImageClassificationDataset, config: MNISTCNNConfiguration):
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

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=config.optimiser,
                  metrics=['accuracy'])
    return model


def cifar_cnn_creator(dataset: ImageClassificationDataset, config: CifarCNNConfiguration):
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
