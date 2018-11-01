import time
import uuid
from pathlib import Path
from uuid import uuid4

import numpy as np
import keras
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.initializers import glorot_normal, constant
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.regularizers import l2

import argparse
import os
import pickle


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class NN:
    def __init__(self, nn_creator, config):
        self._config = config
        self._nn_creator = nn_creator

    def fit(self, dataset):
        init_time = time.time()
        model = self._nn_creator(dataset, self._config)
        results = model.fit(dataset.train_features, dataset.train_targets,
                            validation_data=(dataset.test_features, dataset.test_targets),
                            epochs=self._config.num_updates // self._config.batch_size,
                            batch_size=self._config.batch_size)

        self._model = model
        self._results = results
        self._duration = time.time() - init_time
        return results

    def store(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('mymodel.h5')
        self._model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiemnt took {} min\n'.format(self._duration / 60))
        results_path = path / Path('results.pickle')
        pickle.dump(self._results, open(str(results_path), mode='wb'))


class GPWrapper:
    def __init__(self, gp_creator, config):
        pass

    def fit(self, dataset):
        pass

    def store(self):
        pass


def mnist_cnn_creator(dataset, config):
    assert isinstance(dataset, ImageClassificationDataset)
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


def mnist_fashion_cnn_creator(dataset, config):
    assert isinstance(dataset, ImageClassificationDataset)
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


def cifar_cnn_creator(dataset, config):
    assert isinstance(dataset, ImageClassificationDataset)
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


class grey_cifar10:

    @staticmethod
    def load_data():
        (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
        train_features = rgb2gray(train_features)
        test_features = rgb2gray(test_features)
        return (train_features, train_targets), (test_features, test_targets)


class Dataset:
    def __init__(self, name, train_features, train_targets, test_features, test_targets):
        self._name = name
        self._train_features = train_features
        self._train_targets = train_targets
        self._test_features = test_features
        self._test_targets = test_targets

    @classmethod
    def from_monolitic_data(cls, name, features, targets, test_ratio):
        num_examples = features.shape[0]
        ind = np.random.permutation(range(num_examples))
        train_proportion = int(num_examples * (1 - test_ratio))
        train_features, train_targets = features[ind[:train_proportion]], targets[
            ind[:train_proportion]]
        test_features, test_targets = features[ind[train_proportion:]], targets[
            ind[train_proportion:]]
        return cls(name, train_features, train_targets, test_features, test_targets)

    @classmethod
    def from_keras_format(cls, keras_dataset):
        (train_features, train_targets), (test_features, test_targets) = keras_dataset.load_data()
        name = keras_dataset.__name__.split('.')[-1]
        return cls(name, train_features, train_targets, test_features, test_targets)

    @property
    def train_features(self):
        return self._train_features

    @property
    def train_targets(self):
        return self._train_targets

    @property
    def test_features(self):
        return self._test_features

    @property
    def test_targets(self):
        return self._test_targets

    @property
    def input_shape(self):
        return self._train_features[0].shape

    @property
    def name(self):
        return self._name


class ClassificationDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # we adopt convention to use one hot representation for classes
        if len(self._train_targets.shape) == 1 or self._train_targets.shape[1] == 1:
            num_classes = len(set(self._train_targets.ravel()))
            self._train_targets = keras.utils.to_categorical(self._train_targets, num_classes)
            self._test_targets = keras.utils.to_categorical(self._test_targets, num_classes)

    @property
    def num_classes(self):
        return self._test_targets[0].size


class ImageClassificationDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.input_shape) < 3:  # no channels
            self._train_features = np.expand_dims(self._train_features, axis=-1)
            self._test_features = np.expand_dims(self._test_features, axis=-1)

    @classmethod
    def from_keras_format(cls, keras_dataset):
        (train_features, train_targets), (test_features, test_targets) = keras_dataset.load_data()
        # assuming pixel representation [0,255]
        train_features = train_features / train_features.max()
        test_features = test_features / train_features.max()  # this is correct - preprocessing should not be based on test set
        name = keras_dataset.__name__.split('.')[-1]
        return cls(name, train_features, train_targets, test_features, test_targets)


class Experiment:
    def __init__(self, name, dataset, model):
        self._name = name
        self._model = model
        self._dataset = dataset

    def run(self):
        results_path = Path('/tmp') / Path(str(self) + '-' + str(uuid.uuid4()))
        results = []
        self._model.fit(self._dataset)
        self._model.store(results_path)
        # model_path = Path(results_path) / str(self)
        # model_path.mkdir(parents=True, exist_ok=True)
        return results

    @property
    def name(self):
        return self._name

    def __str__(self):
        return '{}_{}'.format(self._name, self._dataset.name)


class ExperimentSuite:
    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self):
        for experiment in self._experiment_list:
            experiment.run()


class Configuration:

    @classmethod
    def summary(cls):
        summary = ''
        for name, value in cls.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            summary += '_'.join((name, str(value))) + '\n'
        return summary


class MNISTCNNConfiguration(Configuration):
    num_updates = 300
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


class CifarCNNConfiguration(Configuration):
    num_updates = 500
    batch_size = 128
    optimiser = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    @classmethod
    def summary(cls):
        return 'num_updates {}\n' \
               'batch_size {}\n' \
               'optimiser {}'.format(cls.num_updates,
                                     cls.batch_size,
                                     cls.optimiser)


def group_histories(histories):
    validation_loss = []
    test_loss = []
    for h in histories:
        validation_loss.append(h.history['loss'])
        test_loss.append(h.history['val_loss'])
    return np.array(validation_loss), np.array(test_loss)


def main():
    parser = argparse.ArgumentParser(description='Entrypoint for running experiments.')
    parser.add_argument('-datasets', '--datasets', nargs='+',
                        help='The experiments will be executed on these datasets.'
                             'Available: ')
    parser.add_argument('-models', '--models', nargs='+',
                        help='The experiments will be executed on these models.'
                             'Available: ')

    args = parser.parse_args()

    experiment1 = Experiment('cnn_experiment',
                             model=NN(mnist_cnn_creator, config=MNISTCNNConfiguration),
                             dataset=ImageClassificationDataset.from_keras_format(mnist))

    experiment2 = Experiment('cnn_experiment',
                             model=NN(mnist_cnn_creator, config=MNISTCNNConfiguration),
                             dataset=ImageClassificationDataset.from_keras_format(fashion_mnist))

    experiment3 = Experiment('cnn_experiment',
                             model=NN(cifar_cnn_creator, config=CifarCNNConfiguration),
                             dataset=ImageClassificationDataset.from_keras_format(grey_cifar10))

    experiment_suite = ExperimentSuite(experiment_list=[experiment1, experiment2, experiment3])
    experiment_suite.run()

    # histories = experiment.run()
    # train_resutls, test_results = group_histories(histories)
    # plt.plot(np.log(train_resutls.T), np.log(test_results).T, 'b.-', linewidth=0.8)
    # # plt.plot(range(train_resutls[0].size), range(train_resutls[0].size), 'r--')
    # plt.xlabel(r'$\log\ train\ loss$')
    # plt.ylabel(r'$\log\ test\ loss$')
    # plt.show()


if __name__ == '__main__':
    main()
