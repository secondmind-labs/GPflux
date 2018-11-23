import os
import subprocess
import types

import gpflow
import keras
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from gpflux.convolution import PatchHandler, ImagePatchConfig
from refreshed_experiments.data_infrastructure import ImageClassificationDataset, \
    MaxNormalisingPreprocessor


def top1_error(y_true, y_pred):
    return (1 - keras.metrics.categorical_accuracy(y_true, y_pred)) * 100


def top2_error(y_true, y_pred):
    return (1 - keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)) * 100


def top3_error(y_true, y_pred):
    return (1 - keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)) * 100


def get_from_module(name, module):
    if hasattr(module, name):
        return getattr(module, name)
    else:
        available = ' '.join([item for item in dir(module) if
                              not item.startswith('_')])
        raise ValueError('{} not found. Available are {}'.format(name, available))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def group_results(results):
    return np.array([result.history['loss'] for result in results]), \
           np.array([result.history['val_loss'] for result in results])


def labels_onehot_to_int(labels):
    return labels.argmax(axis=-1)[..., None].astype(np.int32)


def reshape_to_2d(x):
    return x.reshape(x.shape[0], -1)


def calc_multiclass_error(model, x, y, batchsize=100):
    Ns = len(x)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(x, splits), np.array_split(y, splits)):
        logits, _ = model.predict_y(xs)
        acc = logits.argmax(1) == ys[:, 0]
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


def calc_avg_nll(model, x, y, batchsize=100):
    num_examples = x.shape[0]
    splits = num_examples // batchsize
    ll = 0
    for xs, ys in zip(np.array_split(x, splits), np.array_split(y, splits)):
        p, _ = model.predict_y(xs)
        p = ((ys == np.arange(10)[None, :]) * p).sum(-1)
        ll += np.log(p).sum()
    ll /= num_examples
    return -ll


class ImagePatchConfigCoder(gpflow.saver.coders.ObjectCoder):
    @classmethod
    def encoding_type(cls):
        return ImagePatchConfig


class PatchHandlerCoder(gpflow.saver.coders.ObjectCoder):
    @classmethod
    def encoding_type(cls):
        return PatchHandler


def save_gpflow_model(filename, model) -> None:
    context = gpflow.SaverContext(coders=[ImagePatchConfigCoder, PatchHandlerCoder])
    gpflow.Saver().save(filename, model, context=context)


def load_gpflow_model(filename, model) -> None:
    context = gpflow.SaverContext(coders=[ImagePatchConfigCoder, PatchHandlerCoder])
    return gpflow.Saver().load(filename, context=context)


def get_top_n_error(model, x, y, n, batchsize=100):
    num_examples = x.shape[0]
    splits = num_examples // batchsize
    hits = 0
    for xs, ys in zip(np.array_split(x, splits), np.array_split(y, splits)):
        p, _ = model.predict_y(xs)
        hits += (ys == p.argsort(-1)[:, -n:]).sum()
    return 1 - hits / num_examples


def get_avg_nll_missclassified(model, x, y, batchsize=100):
    num_examples = x.shape[0]
    splits = num_examples // batchsize
    ll = 0
    num_missclassified = 0
    for xs, ys in zip(np.array_split(x, splits), np.array_split(y, splits)):
        p, _ = model.predict_y(xs)
        full_p = ((ys == np.arange(10)[None, :]) * p).sum(-1)  # (batchsize,)
        # we need to keep only missclassified
        missclassified_ind = (ys != p.argmax(-1)[..., None]).ravel()
        num_missclassified += np.sum(missclassified_ind)
        missclassified_p = full_p[missclassified_ind]
        ll += np.log(missclassified_p).sum()
    ll /= num_missclassified
    return -ll


def name_to_summary(name):
    _, trainer, config, creator, dataset, *_ = name.split('-')
    return 'Trainer: {}\nConfig: {}\nCreator: {}\nDataset: {}\n'.format(trainer, config, creator,
                                                                        dataset)


def calc_ece_from_probs(probs, labels, n_bins=10):
    # labels are in int representation
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = np.argmax(probs, axis=1).ravel()
    confidences = np.array(
        [probs[i, pred] for i, pred in enumerate(predictions)])
    accuracies = (labels.ravel() == predictions)

    _ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (bin_lower < confidences) & (confidences <= bin_upper)

        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            _ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            pass

    return _ece


def calculate_ece_score(model, x, y, batchsize=100):
    num_examples = x.shape[0]
    splits = num_examples // batchsize
    probs = []
    for xs in np.array_split(x, splits):
        p, _ = model.predict_y(xs)
        probs.append(p)
    probs = np.concatenate(probs, axis=0)

    return calc_ece_from_probs(probs, y)


def _get_max_normalised(data, name):
    (train_features, train_targets), (test_features, test_targets) = data
    dataset = \
        ImageClassificationDataset.from_train_test_split(name,
                                                         train_features=train_features,
                                                         train_targets=train_targets,
                                                         test_features=test_features,
                                                         test_targets=test_targets)
    return MaxNormalisingPreprocessor.preprocess(dataset)


def _mix_train_test(data, random_state):
    (train_features, train_targets), (test_features, test_targets) = data
    ratio = len(train_targets) / (len(train_targets) + len(train_features))
    features = np.concatenate((train_features, test_features), axis=0)
    targets = np.concatenate((train_targets, test_targets), axis=0)
    train_features, test_features, train_targets, test_targets = \
        train_test_split(features,
                         targets,
                         test_size=ratio,
                         random_state=random_state)
    return (train_features, train_targets), (test_features, test_targets)


def load_svhn():
    if not os.path.exists('/tmp/svhn_train.mat'):
        subprocess.call(
            ["wget", "-O", "/tmp/svhn_train.mat",
             "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"])
    if not os.path.exists('/tmp/svhn_test.mat'):
        subprocess.call(
            ["wget", "-O", "/tmp/svhn_test.mat",
             "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"])
    train_data = loadmat('/tmp/svhn_train.mat')
    test_data = loadmat('/tmp/svhn_test.mat')
    x_train, y_train = train_data['X'], train_data['y']
    x_test, y_test = test_data['X'], test_data['y']
    x_train, x_test = np.transpose(x_train, [3, 0, 1, 2]), np.transpose(x_test, [3, 0, 1, 2])
    x_train, x_test = rgb2gray(x_train), rgb2gray(x_test)
    data = (x_train, y_train.ravel()-1), (x_test, y_test.ravel()-1)
    return data

def load_grey_cifar():
    from keras.datasets import cifar10
    (train_features, train_targets), (test_features, test_targets) = cifar10.load_data()
    train_features = rgb2gray(train_features)
    test_features = rgb2gray(test_features)
    return (train_features, train_targets), (test_features, test_targets)
