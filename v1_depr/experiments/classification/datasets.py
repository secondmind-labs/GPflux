from pathlib import Path
from typing import Tuple, List

import gpflow
import numpy as np
import observations as obs
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image_process


DatasetTrain = Tuple[np.ndarray, np.ndarray]
DatasetTest = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[DatasetTrain, DatasetTest]


def rotmnist_data(tmp_dir) -> Dataset:
    url = 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip'
    obs.maybe_download_and_extract(tmp_dir, url, extract=True)

    def data_for(pattern):
        candidate = [f for f in Path(tmp_dir).iterdir() if str(f).endswith(pattern)]
        if not candidate:
            raise FileNotFoundError(f'Pattern {pattern} not found.')
        elif len(candidate) > 1:
            raise FileNotFoundError(f'Ambiguous pattern {pattern} passed.')
        return candidate[0]

    test_file = data_for('test.amat')
    train_file = data_for('train_valid.amat')

    test_data = np.loadtxt(test_file, delimiter=' ')
    train_data = np.loadtxt(train_file, delimiter=' ')

    test_x, test_y = test_data[..., :-1], test_data[..., -1].astype(np.uint8)
    train_x, train_y = train_data[..., :-1], train_data[..., -1].astype(np.uint8)

    return (train_x, train_y), (test_x, test_y)


def input_shape(dataset_name: str) -> List[int]:
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return [32, 32, 3]
    elif dataset_name == "mnist" or dataset_name == "rotmnist":
        return [28, 28, 1]


def get_dataset(dataset_name: str, standartize: bool = True, x_type=None, y_type=None) -> Dataset:
    tmp_dir = Path('~/.datasets', dataset_name).expanduser()

    dataset_getter = dict(
        mnist=mnist_data,
        rotmnist=rotmnist_data,
        cifar10=cifar10_data
    )[dataset_name]

    (x, y), (xt, yt) = dataset_getter(tmp_dir)

    def upd_y(d):
        return d.reshape((-1, 1))

    y, yt = upd_y(y), upd_y(yt)

    def astype(inputs, dtype):
        return [i.astype(dtype) for i in inputs]

    int_type = gpflow.settings.int_type
    float_type = gpflow.settings.float_type

    x_type = float_type if x_type is None else x_type
    y_type = int_type if y_type is None else y_type
    x, xt = astype([x, xt], x_type)
    y, yt = astype([y, yt], y_type)
    return (x, y), (xt, yt)


def standardise(data):
    (x, y), (xt, yt) = data
    return (x.astype(np.float64) / 255., y), (xt.astype(np.float64) / 255., y)


def mnist_data(tmp_dir) -> Dataset:
    return standardise(obs.mnist(tmp_dir))


def cifar10_data(tmp_dir):
    (x, y), (xt, yt) = obs.cifar10(tmp_dir)

    def upd_x(d):
        axes = (0, 2, 3, 1)
        return np.transpose(d, axes)

    data = (upd_x(x), y), (upd_x(xt), yt)
    return standardise(data)


def augment_image(img, augment_factor=1, use_rotation=False, use_shear=False, use_shift=False, use_zoom=False):
    augments = [use_rotation and 'rotate',
                use_shear and 'shear',
                use_shift and 'shift',
                use_zoom and 'zoom']
    augments = [a for a in augments if a != False]
    for i in range(0, augment_factor):
        row_col_channels = dict(row_axis=0, col_axis=1, channel_axis=2)
        choice = np.random.choice(augments)
        if choice == 'rotate':
            img = image_process.random_rotation(img, 30, **row_col_channels)
        elif choice == 'shear':
            img = image_process.random_shear(img, 0.2, **row_col_channels)
        elif choice == 'shift':
            img = image_process.random_shift(img, 0.2, 0.2, **row_col_channels)
        elif choice == 'zoom':
            img = image_process.random_zoom(img, 0.9, **row_col_channels)
    return img


class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=128, seed=None, shuffle=True, augment_cb=None):
        self.x = x
        self.y = y
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(x.shape[0])
        self.augment_cb = augment_cb
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indices = self.indices[start:end]
        x_batch = self.x[indices, ...]
        y_batch = self.y[indices, ...]
        if self.augment_cb is not None:
            for i in range(x_batch.shape[0]):
                x_batch[i] = self.augment_cb(x_batch[i])
        return x_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.rng.shuffle(self.indices)