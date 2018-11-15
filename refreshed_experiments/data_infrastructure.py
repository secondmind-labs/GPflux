import keras
import numpy as np


class Dataset:
    def __init__(self, name: str,
                 train_features: np.ndarray,
                 train_targets: np.ndarray,
                 test_features: np.ndarray,
                 test_targets: np.ndarray):
        # feature shape (num_examples, *data_shape) and targets shape (num_examples, *targets_shape)
        self._name = name
        self._train_features = train_features
        self._train_targets = train_targets
        self._test_features = test_features
        self._test_targets = test_targets

    @classmethod
    def from_monolitic_data(cls,
                            name: str,
                            features: np.ndarray,
                            targets: np.ndarray,
                            test_ratio: float):
        num_examples = features.shape[0]
        ind = np.random.permutation(range(num_examples))
        num_train = int(num_examples * (1 - test_ratio))
        train_features, train_targets = features[ind[:num_train]], targets[ind[:num_train]]
        test_features, test_targets = features[ind[num_train:]], targets[ind[num_train:]]
        return cls(name, train_features, train_targets, test_features, test_targets)

    @classmethod
    def from_keras_format(cls,keras_dataset):
        (train_features, train_targets), (test_features, test_targets) = keras_dataset.load_data()
        name = keras_dataset.__name__.split('.')[-1]
        return cls(name, train_features, train_targets, test_features, test_targets)

    @classmethod
    def from_train_test_split(cls,
                              name: str,
                              train_features: np.ndarray,
                              train_targets: np.ndarray,
                              test_features: np.ndarray,
                              test_targets: np.ndarray):
        return cls(name, train_features, train_targets, test_features, test_targets)

    @property
    def train_features(self) -> np.ndarray:
        return self._train_features

    @property
    def train_targets(self) -> np.ndarray:
        return self._train_targets

    @property
    def test_features(self) -> np.ndarray:
        return self._test_features

    @property
    def test_targets(self) -> np.ndarray:
        return self._test_targets

    @property
    def input_shape(self) -> np.ndarray:
        return self._train_features[0].shape

    @property
    def name(self) -> str:
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
    def num_classes(self) -> int:
        return self._test_targets[0].size


class ImageClassificationDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        # assumes that we represent images as (num_examples, width, height, num_channels)
        super().__init__(*args, **kwargs)
        if len(self.input_shape) < 3:  # no channels
            self._train_features = np.expand_dims(self._train_features, axis=-1)
            self._test_features = np.expand_dims(self._test_features, axis=-1)

    @classmethod
    def from_keras_format(cls, keras_dataset) -> 'ImageClassificationDataset':
        (train_features, train_targets), (test_features, test_targets) = keras_dataset.load_data()
        name = keras_dataset.__name__.split('.')[-1]
        return cls(name, train_features, train_targets, test_features, test_targets)


class DatasetPreprocessor:
    @staticmethod
    def preprocess(dataset: Dataset) -> Dataset:
        raise NotImplementedError()


class DummyPreprocessor(DatasetPreprocessor):
    @staticmethod
    def preprocess(dataset: Dataset) -> Dataset:
        return dataset


class MaxNormalisingPreprocessor(DatasetPreprocessor):
    @staticmethod
    def preprocess(dataset: Dataset) -> Dataset:
        train_features, test_features = dataset.train_features, dataset.test_features
        _max = train_features.max()  # we infer max only from train data
        train_features = train_features / _max
        test_features = test_features / _max
        return dataset.from_train_test_split(dataset.name + '_max_normalised',
                                             train_features,
                                             dataset.train_targets,
                                             test_features,
                                             dataset.test_targets)
