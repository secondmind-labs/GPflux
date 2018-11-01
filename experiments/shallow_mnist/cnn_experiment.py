import numpy as np
import keras
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.initializers import glorot_normal, constant
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.regularizers import l2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def mnist_cnn_creator(dataset, cnn_configuration):
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
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def mnist_fashion_cnn_creator(dataset, cnn_configuration):
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
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def cifar_cnn_creator(dataset, cnn_configuration):
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

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
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
    def __init__(self, train_features, train_targets, test_features, test_targets):
        # assumes dense representation of the classes, i.e. (num_examples, 1) shape of targets
        self._train_features = train_features
        self._train_targets = train_targets
        self._test_features = test_features
        self._test_targets = test_targets

    @classmethod
    def from_monolitic_data(cls, features, targets, test_ratio):
        num_examples = features.shape[0]
        ind = np.random.permutation(range(num_examples))
        train_proportion = int(num_examples * (1 - test_ratio))
        train_features, train_targets = features[ind[:train_proportion]], targets[
            ind[:train_proportion]]
        test_features, test_targets = features[ind[train_proportion:]], targets[
            ind[train_proportion:]]
        return cls(train_features, train_targets, test_features, test_targets)

    @classmethod
    def from_keras_format(cls, keras_dataset):
        (train_features, train_targets), (test_features, test_targets) = keras_dataset.load_data()
        return cls(train_features, train_targets, test_features, test_targets)

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
        train_features = train_features / train_features.max()
        test_features = test_features / train_features.max()  # this is correct - preprocessing should not be based on test set
        return cls(train_features, train_targets, test_features, test_targets)


class Experiment:
    def __init__(self, name, model_creator, model_configuration, dataset, experiment_configuration):
        self._name = name
        self._model_creator = model_creator
        self._model_configuration = model_configuration
        self._dataset = dataset
        self._experiment_configuration = experiment_configuration

    def run(self):
        results = []
        for _ in range(self._experiment_configuration.num_repetitions):
            model = self._model_creator(self._dataset, self._model_configuration)
            history = model.fit(self._dataset.train_features, self._dataset.train_targets,
                                validation_data=(
                                    self._dataset.test_features, self._dataset.test_targets),
                                epochs=self._model_configuration.num_updates // self._model_configuration.batch_size,
                                batch_size=self._model_configuration.batch_size)
            results.append(history)
        return results

    @property
    def name(self):
        return self._name


class ExperimentSuite:
    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self):
        results_dict = {}
        for experiment in self._experiment_list:
            results_dict[experiment.name] = experiment.run()
        return results_dict


class ExperimentConfiguration:
    num_repetitions = 1


class MNISTCNNConfiguration:
    num_updates = 3000
    batch_size = 128


class CifarCNNConfiguration:
    num_updates = 3000
    batch_size = 128


def group_histories(histories):
    validation_loss = []
    test_loss = []
    for h in histories:
        validation_loss.append(h.history['loss'])
        test_loss.append(h.history['val_loss'])
    return np.array(validation_loss), np.array(test_loss)


def main():
    experiment1 = Experiment('mnist_nn_experiment',
                             model_creator=mnist_cnn_creator,
                             model_configuration=MNISTCNNConfiguration,
                             dataset=ImageClassificationDataset.from_keras_format(mnist),
                             experiment_configuration=ExperimentConfiguration)

    experiment2 = Experiment('mnist_fahsion_nn_experiment',
                             model_creator=mnist_fashion_cnn_creator,
                             model_configuration=MNISTCNNConfiguration,
                             dataset=ImageClassificationDataset.from_keras_format(fashion_mnist),
                             experiment_configuration=ExperimentConfiguration)

    experiment3 = Experiment('cifar10_nn_experiment',
                             model_creator=cifar_cnn_creator,
                             model_configuration=CifarCNNConfiguration,
                             dataset=ImageClassificationDataset.from_keras_format(grey_cifar10),
                             experiment_configuration=ExperimentConfiguration)

    experiment_suite = ExperimentSuite(experiment_list=[experiment3])
    experiment_suite.run()

    # histories = experiment.run()
    # train_resutls, test_results = group_histories(histories)
    # plt.plot(np.log(train_resutls.T), np.log(test_results).T, 'b.-', linewidth=0.8)
    # # plt.plot(range(train_resutls[0].size), range(train_resutls[0].size), 'r--')
    # plt.xlabel(r'$\log\ train\ loss$')
    # plt.ylabel(r'$\log\ test\ loss$')
    # plt.show()


main()
