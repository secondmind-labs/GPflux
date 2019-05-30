from pathlib import Path
from typing import Union

import tensorflow as tf
import numpy as np
import observations as obs
import pandas as pd

import gpflow
from gpflux.convolution.convolution_utils import ImagePatchConfig, PatchHandler


def load_semeion_dataset(cache_dir):
    cache_filename = Path(cache_dir, "semeion.npz")
    filename = str(cache_filename)
    if cache_filename.exists():
        _, data = zip(*np.load(filename).items())
        return data, data
    semeion_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'
    d = pd.read_csv(semeion_url, header=None, delim_whitespace=True)
    d = np.array(d)
    orig_size = 16
    dest_size = 28
    images = d[:, :-10].reshape(-1, orig_size, orig_size, 1)
    classes = np.argmax(d[:, -10:], axis=1)
    with tf.Session(graph=tf.Graph()) as session:
      images_tf = tf.image.resize_image_with_crop_or_pad(images, dest_size, dest_size)
      images = session.run(images_tf).reshape(-1, dest_size * dest_size)
      data = images * 255, classes
      np.savez(filename, *data)
      return data, data


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_dataset(dataset: str):  # dataset = [mnist, mnist01, cifar]
    def general_preprocess(X, Y, Xs, Ys):
        Y = Y.astype(int)
        Ys = Ys.astype(int)
        Y = Y.reshape(-1, 1)
        Ys = Ys.reshape(-1, 1)
        return X, Y, Xs, Ys

    def preprocess_mnist_ij(X, Y, Xs, Ys, i, j):
        def filter_ij(X, Y):
            lbls_ij = np.logical_or(Y == i, Y == j).flatten()
            return X[lbls_ij, :], Y[lbls_ij, :]

        X, Y = filter_ij(X, Y)
        Xs, Ys = filter_ij(Xs, Ys)
        # set labels to 0 for i and 1 for j
        Y[(Y == i).flatten()] = 0
        Ys[(Ys == i).flatten()] = 0
        Y[(Y == j).flatten()] = 1
        Ys[(Ys == j).flatten()] = 1
        return X, Y, Xs, Ys

    def preprocess_cifar(X, Y, Xs, Ys):
        X = np.transpose(X, [0, 2, 3, 1])
        Xs = np.transpose(Xs, [0, 2, 3, 1])
        X = X.reshape(-1, (32 ** 2) * 3)
        Xs = Xs.reshape(-1, (32 ** 2) * 3)
        return X, Y, Xs, Ys

    def preprocess_full(X, Y, Xs, Ys):
        return X, Y, Xs, Ys

    def preprocess_semeion(X, Y, Xs, Ys):
        return X, Y, Xs, Ys

    def preprocess_svhn(X, Y, Xs, Ys):
        X = rgb2gray(X)
        Xs = rgb2gray(Xs)
        with tf.Session(graph=tf.Graph()) as sess:
            X_resize = tf.image.resize_images(X[..., None], size=(28, 28))
            Xs_resize = tf.image.resize_images(Xs[..., None], size=(28, 28))
            X, Xs = sess.run([X_resize, Xs_resize])
        Y = Y.astype(int)
        Ys = Ys.astype(int)
        Y = Y.reshape(-1, 1)
        Ys = Ys.reshape(-1, 1)
        D = 28 * 28
        X = X.reshape(-1, D)
        Xs = Xs.reshape(-1, D)
        return X, Y, Xs, Ys

    data_dict = dict(
        mnist01=obs.mnist, 
        mnist27=obs.mnist, 
        mnist=obs.mnist,
        cifar=obs.cifar10, 
        svhn=obs.svhn,
        semeion=load_semeion_dataset)

    preprocess_dict = dict(
        mnist01=lambda X, Y, Xs, Ys: preprocess_mnist_ij(X, Y, Xs, Ys, 0, 1),
        mnist27=lambda X, Y, Xs, Ys: preprocess_mnist_ij(X, Y, Xs, Ys, 2, 7),
        mnist=preprocess_full,
        cifar=preprocess_cifar,
        semeion=preprocess_semeion,
        svhn=preprocess_svhn)

    data_func = data_dict[dataset]

    dataset_path = Path("~/.datasets/").expanduser()
    (X, Y), (Xs, Ys) = data_func(dataset_path)
    X, Y, Xs, Ys = general_preprocess(X, Y, Xs, Ys)
    preprocess_func = preprocess_dict[dataset]
    X, Y, Xs, Ys = preprocess_func(X, Y, Xs, Ys)

    alpha = 255.0
    return (X / alpha, Y), (Xs / alpha, Ys)


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



def load_gpflow_model(filename) -> gpflow.models.Model:
    context = gpflow.SaverContext(coders=[ImagePatchConfigCoder, PatchHandlerCoder])
    return gpflow.Saver().load(filename, context=context)



def plot(images):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, ...])
    plt.show()

def get_miclassified_binary(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = ((p > 0.5).astype('float') == ys)
        hits.append(acc)
    return np.concatenate(hits, 0)

def calc_binary_error(model, Xs, Ys, batchsize=100, **kwargs):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = ((p > 0.5).astype('float') == ys)
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


def calc_multiclass_error(model, Xs, Ys, batchsize=100, mc=1):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        if mc > 1:
            N_original = xs.shape[0]
            xs = np.tile(xs[None, ...], [mc, 1, 1]).reshape(N_original * mc, -1)
        p, _ = model.predict_y(xs)
        if mc > 1:
            p = p.reshape(mc, N_original, -1)
            p = p.mean(axis=0)
        acc = p.argmax(1) == ys[:, 0]
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


def get_error_cb(model, Xs, Ys, error_func, full=False, Ns=250, mc=1):
    def error_cb(*args, **kwargs):
        if full:
            xs, ys = Xs, Ys
        else:
            xs, ys = Xs[:Ns], Ys[:Ns]
        return error_func(model, xs, ys, batchsize=32, mc=1)
    return error_cb


def trace(T, sess, name):
    import tensorflow as tf
    from tensorflow.python.client import timeline


    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(T, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(name, 'w') as f:
        f.write(chrome_trace)



def compute_predictions(model, data, mc_samples=10, batch_size=100, num_classes=10):
    batch_size //= mc_samples
    Xs, Ys = data
    Nc = num_classes
    Ns = Xs.shape[0]
    splits = Ns // batch_size + int(Ns <= batch_size)
    indices = np.array_split(np.arange(Ns), splits)
    Xs_chunks = np.array_split(Xs, splits)
    Ys_chunks = np.array_split(Ys, splits)

    misses = [None] * len(indices)
    i = 0
    for idx, xs, ys in zip(indices, Xs_chunks, Ys_chunks):
        num = mc_samples
        ns = len(xs)
        D = Xs.shape[-1]
        xss = np.tile(xs[None, ...], [mc_samples, 1, 1])  # [mc_samples, ns, D]
        xss = np.reshape(xss, [mc_samples * ns, D])  # [mc_samples * ns, D]
        probs, _ = model.predict_y(xss)  # num*ns x Nc
        probs = np.reshape(probs, [mc_samples, ns, Nc])  # [mc_samples, ns, Nc]
        probs = np.mean(probs, axis=0, keepdims=False)  # [ns, Nc]

        missed = np.argmax(probs, axis=1) != ys[:, 0]
        misses[i] = (idx[missed], xs[missed], ys[missed], probs[missed], probs)
        i += 1

    def concat(arr):
        return np.concatenate(arr, axis=0)

    idx, xs_miss, ys_miss_true, probs_miss, probs = tuple(map(concat, zip(*misses)))
    return dict(
        idx=idx,
        xs_miss=xs_miss,
        ys_miss_true=ys_miss_true,
        probs_miss=probs_miss,
        probs=probs,
    )


