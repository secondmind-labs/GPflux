import keras
import numpy as np


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


class DatasetPreprocessor:
    @staticmethod
    def preprocess(dataset):
        raise NotImplementedError()


class DummyPreprocessor(DatasetPreprocessor):
    @staticmethod
    def preprocess(dataset):
        return dataset


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