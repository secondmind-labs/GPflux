from pathlib import Path
from typing import Union

import numpy as np
import observations as obs
import pandas as pd
import tensorflow as tf

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

    def preprocess_mnist01(X, Y, Xs, Ys):
        def filter_01(X, Y):
            lbls01 = np.logical_or(Y == 0, Y == 1).flatten()
            return X[lbls01, :], Y[lbls01, :]

        X, Y = filter_01(X, Y)
        Xs, Ys = filter_01(Xs, Ys)
        return X, Y, Xs, Ys

    def preprocess_cifar(X, Y, Xs, Ys):
        X = rgb2gray(np.transpose(X, [0, 2, 3, 1]))
        Xs = rgb2gray(np.transpose(Xs, [0, 2, 3, 1]))
        X = X.reshape(-1, 32 ** 2)
        Xs = Xs.reshape(-1, 32 ** 2)
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
        mnist01=obs.mnist, mnist=obs.mnist,
        cifar=obs.cifar10, svhn=obs.svhn,
        semeion=load_semeion_dataset)

    preprocess_dict = dict(
        mnist01=preprocess_mnist01,
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


def calc_binary_error(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = ((p > 0.5).astype('float') == ys)
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


def calc_multiclass_error(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = p.argmax(1) == ys[:, 0]
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


def get_error_cb(model, Xs, Ys, error_func, full=False, Ns=500):
    def error_cb(*args, **kwargs):
        if full:
            xs, ys = Xs, Ys
        else:
            xs, ys = Xs[:Ns], Ys[:Ns]
        return error_func(model, xs, ys, batchsize=50)
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
