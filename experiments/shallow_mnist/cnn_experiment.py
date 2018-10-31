import numpy as np
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def cnn_creator(dataset, cnn_configuration):
    assert isinstance(dataset, ClassificationDataset)
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


class Dataset:
    def __init__(self, train_features, train_targets, test_features, test_targets):
        # assumes dense representation of the classes, i.e. (num_examples, 1) shape of targets
        self._train_features = train_features
        self._train_targets = train_targets
        self._test_features = test_features
        self._test_targets = test_targets

    @staticmethod
    def from_monolitic_data(features, targets, test_ratio):
        num_examples = features.shape[0]
        ind = np.random.permutation(range(num_examples))
        train_proportion = int(num_examples * (1 - test_ratio))
        train_features, train_targets = features[ind[:train_proportion]], targets[
            ind[:train_proportion]]
        test_features, test_targets = features[ind[train_proportion:]], targets[
            ind[train_proportion:]]
        return Dataset(train_features, train_targets, test_features, test_targets)

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

    @property
    def num_classes(self):
        return self._test_targets[0].size


class Experiment:
    def __init__(self, model_creator, model_configuration, dataset, experiment_configuration):
        self._model_creator = model_creator
        self._model_configuration = model_configuration
        self._dataset = dataset
        self._experiment_configuration = experiment_configuration

    def run(self):
        histories = []
        for _ in range(self._experiment_configuration.num_repetitions):
            model = self._model_creator(self._dataset, self._model_configuration)
            history = model.fit(self._dataset.train_features, self._dataset.train_targets,
                                validation_data=(
                                self._dataset.test_features, self._dataset.test_targets),
                                epochs=5)
            histories.append(history)
        return histories


class ExperimentConfiguration:
    num_repetitions = 2


def group_histories(histories):
    ll = []
    for h in histories:
        ll.append(h.history['val_loss'])
    return np.array(ll)


def plot_results(results_dict):
    pass


def main():
    model_creator = cnn_creator
    model_configuration = None
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if len(x_train.shape) == 3:
        x_train, x_test = x_train.expand_dims(-1), x_test.expand_dims(-1)  # no channels

    num_classes = len(set(y_test.ravel()))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    dataset = ClassificationDataset(x_train, y_train, x_test, y_test)
    experiment = Experiment(model_creator, model_configuration, dataset, ExperimentConfiguration)
    histories = experiment.run()
    results = group_histories(histories)
    print(results)


main()
