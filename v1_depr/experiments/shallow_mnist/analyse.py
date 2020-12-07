from collections import OrderedDict
from functools import reduce
from pathlib import Path
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sacred import Experiment
from utils import get_dataset, load_gpflow_model

import gpflow

NAME = "analyze-models"
ex = Experiment(NAME)


@ex.config
def config():
    dataset = "mnist"      # mnist | mnist01 | cifar10 | semeion | svhn
    batch_size = 100
    figure_format = "pdf"

    models = {
        "CNN": {
            "filename": "~/experiments/mnist/cnn_27-17-23_lr-0.0001_batchsize-128/cnn.90-0.99.h5",
            "type": "cnn"
        },
        "ConvGP": {
            "filename": "~/experiments/mnist/convgp_26-19-50_W-True_Ti-False_initpatches-patches-unique_kern-RBF_lr-0.005_lrdecay-custom_nip-1000_batchsize-128_patch-5/convgp.gpflow",
            "type": "convgp",
            "mc_samples": 10
        },
        "TICK-GP": {
            "filename": "~/experiments/mnist/mnist_W_True_I_True_init_patches_patches-unique_kern_RBF_adam_decay_M_1000_minibatch_size_150_patch_5.gpflow",
            "type": "convgp",
            "mc_samples": 10
        },
    }

    figures_dir = "./figures"
    cache_dir = "./cache/"
    bar_height_limit = 0.01
    max_images_to_compare = 8  # Option [10, *list(range(7))]
    figure_compare_filename = "compare_models_grid"

    use_cache = True
    plot = {
        'missings': True,
        'compare': True
    }


@ex.capture
def figure_filename(name, figures_dir, figure_format):
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return str(Path(fig_dir, f"{name}.{figure_format}"))


@ex.capture
def cache_filename(name, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(Path(cache_dir, f"{name}.npz"))


@ex.capture
def cache_file_exists(name, use_cache):
    if not use_cache:
        return False
    return Path(cache_filename(name)).exists()


def cache_save_data(name, data):
    filename = cache_filename(name)
    np.savez(filename, data)


def cache_load_data(name):
    npz = np.load(cache_filename(name))
    _, (data,) = zip(*npz.items())
    return data


@ex.capture
def num_classes(dataset):
    if dataset in ["mnist", "cifar10", "semeion", "svhn"]:
        return 10


@ex.capture
def get_test_dataset(model_type, dataset) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys = get_dataset(dataset)[1]
    if model_type == "cnn" and dataset in ["mnist", "semeion", "svhn"]:
        Xs = Xs.reshape(-1, 28, 28, 1)
    return Xs, Ys


@ex.capture
def get_misses(model, data, name, config, batch_size):
    kwargs = dict(batch_size=batch_size)
    mc_samples = config.get('mc_samples')
    if mc_samples is not None:
        kwargs['mc_samples'] = mc_samples
    results = compute_misses(model, data, config.get('type'), **kwargs)
    cache_save_data(name, results)
    return results


def compute_misses(model, data, model_type, mc_samples=10, batch_size=100):
    if model_type == 'convgp':
        batch_size /= mc_samples
    Xs, Ys = data
    Nc = num_classes()
    Ns = Xs.shape[0]
    splits = Ns // batch_size + int(Ns <= batch_size)
    indices = np.array_split(np.arange(Ns), splits)
    Xs_chunks = np.array_split(Xs, splits)
    Ys_chunks = np.array_split(Ys, splits)

    misses = []
    for idx, xs, ys in zip(indices, Xs_chunks, Ys_chunks):
        num = mc_samples
        ns = len(xs)
        #  xs: Ns x D
        if model_type == 'convgp':
            D = Xs.shape[-1]
            xss = np.tile(xs[None, ...], [mc_samples, 1, 1])  # mc_samples x ns x D
            xss = np.reshape(xss, [mc_samples * ns, D])  # mc_samples*ns x D
            probs, _ = model.predict_y(xss)  # num*ns x Nc
            probs = np.reshape(probs, [mc_samples, ns, Nc])  # mc_samples x ns x Nc
            probs = np.mean(probs, axis=0, keepdims=False) # ns x Nc
        else:
            probs = model.predict(xs)

        missed = np.argmax(probs, axis=1) != ys[:, 0]
        misses.append((xs[missed], ys[missed], probs[missed], probs, idx[missed]))

    def concat(arr):
        return np.concatenate(arr, axis=0)

    return tuple(map(concat, zip(*misses)))


def top_n_errors(n_errors, Pm, Ym, total_dataset_size):
    if isinstance(n_errors, int):
        n_errors = [n_errors]
    errs = OrderedDict()
    for n in n_errors:
        sorted_n_indices = np.argsort(-Pm, axis=1)[:, :n]
        misses = [ym not in pm_indices for pm_indices, ym  in zip(sorted_n_indices, Ym)]
        misses = np.array(misses).sum()
        errs[n] = (misses / total_dataset_size) * 100
    return errs


def miss_probs(Pm, Ym_true):
    return np.array([Pm[i, item] for i, item in enumerate(Ym_true)])


def log_likelihood(Pm, Ym_true):
    jitter = 1e-10
    return np.mean(np.log(miss_probs(Pm, Ym_true) + jitter))


def spines_color(ax, color='gray'):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)

def spines_visibility(ax, visible=False):
    [spine.set_visible(visible) for spine in ax.spines.values()]


def plot_ax_prob_bars(ax, hit_id, miss_id, probs, hit_color="C0", miss_color="C1", visible_spines=None):
    bars = ax.bar(np.arange(num_classes()), probs, color=miss_color)
    ax.set_ylim([0, 1.3])

    if visible_spines is None:
        spines_visibility(ax, True)
    else:
        spines_visibility(ax, False)
        [ax.spines[pos].set_visible(True) for pos in visible_spines]

    miss_bar = bars[miss_id]
    hit_bar = bars[hit_id]

    miss_bar.set_color(miss_color)
    hit_bar.set_color(hit_color)

    def draw_index(bar, idx, color):
        h, w = bar.get_height(), bar.get_width()
        ax.text(bar.get_x() + w / 2., 1.01 * h, f"{idx}", ha='center', va='bottom', color=color)

    draw_index(miss_bar, miss_id, miss_color)
    draw_index(hit_bar, hit_id, hit_color)


@ex.capture
def plot_missed_images(Xm, Pm, Ym, filename, bar_height_limit):
    Pt = miss_probs(Pm, Ym)
    p_diff = ((np.max(Pm, axis=1)).reshape(-1, 1) - Pt).flatten()
    sorted_indices = np.argsort(p_diff)

    Xm = Xm[sorted_indices]
    Pm = Pm[sorted_indices]
    Ym = Ym[sorted_indices]

    n_xm = Xm.shape[0]
    n_plot_cols = 10
    n_plot_rows = n_xm // n_plot_cols + int((n_xm % n_plot_cols) > 0)
    fig, axes = plt.subplots(n_plot_rows, n_plot_cols * 2, figsize=(18, n_plot_rows))
    axes = axes.flatten()

    for ax in axes:
        spines_visibility(ax, False)
        spines_color(ax)
        ax.set_yticks([])
        ax.set_xticks([])

    for i, (xm, pm, ys) in enumerate(zip(Xm, Pm, Ym)):
        miss_id = np.argmax(pm)
        hit_id = ys[0]
        ax1, ax2 = axes[2*i], axes[2*i + 1]
        ax1.imshow(xm.reshape(28, 28), cmap='Blues')
        spines_visibility(ax1, True)
        plot_ax_prob_bars(ax2, hit_id, miss_id, pm, visible_spines=['bottom'])

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)

    plt.show()


def plot_ims(inter_missing_info):
    first = list(inter_missing_info.keys())[0]
    cols = len(inter_missing_info[first][0])
    fig1 = plt.figure()
    gs1 = plt.GridSpec(3, cols, figure=fig1)
    for i, name in enumerate(inter_missing_info):
        for c in range(cols):
            ax = plt.subplot(gs1[i, c])
            ax.imshow(inter_missing_info[name][0][c].reshape(28, 28), cmap="Blues_r")

    plt.show()


def plot_image_compare_grid(inter_missing_info, filename=None):
    first = list(inter_missing_info.keys())[0]
    rows = len(inter_missing_info) + 1
    cols = len(inter_missing_info[first][0])
    fig = plt.figure()
    height_ratios = [1] + [1.3] * (rows - 1)
    gs = plt.GridSpec(rows, cols, height_ratios=height_ratios, figure=fig)

    # plot_ims(inter_missing_info)

    def reset_ticks(ax):
        ax.set_yticks([])
        ax.set_xticks([])

    for c in range(cols):
        img = inter_missing_info[first][0][c]
        ax = plt.subplot(gs[0, c])
        ax.imshow(img.reshape(28, 28), cmap='Blues')
        spines_color(ax)
        reset_ticks(ax)

    for r, name in zip(range(1, rows), inter_missing_info.keys()):
        for c in range(cols):
            ax = plt.subplot(gs[r, c])
            spines_color(ax)
            reset_ticks(ax)
            if c == 0:
                ax.set_ylabel(name)
            _, Ym, Pm, _, _ = inter_missing_info[name]
            ym, pm = Ym[c], Pm[c]
            miss_id = np.argmax(pm)
            hit_id = ym[0]
            plot_ax_prob_bars(ax, hit_id, miss_id, pm)

    if filename is not None:
        fig.savefig(filename)

    plt.show()


def load_model(filename, model_type):
    filename = str(Path(filename).expanduser())
    if model_type == "cnn":
        return tf.keras.models.load_model(filename)
    elif model_type == "convgp":
        model = load_gpflow_model(filename)
        model.compile()
        return model


@ex.capture
def compare_models(missings_info, max_images_to_compare, figure_compare_filename):
    sets = [info[-1].astype(np.int32) for _, info in missings_info.items()]
    intersections = reduce(np.intersect1d, sets)

    if isinstance(max_images_to_compare, (list, tuple, np.ndarray)):
        intersections = intersections[max_images_to_compare]
    elif isinstance(max_images_to_compare, int):
        intersections = intersections[:max_images_to_compare]
    else:
        raise ValueError(f'Unknown type for max_images_to_compare {type(max_images_to_compare)}')

    def select_itersections(info, indices):
        return [e[indices, ...] for e in info]

    inter_missing_info = OrderedDict()
    for inter_name, inter_info in missings_info.items():
        ids = np.array([np.argwhere(inter_info[-1] == i).reshape(()) for i in intersections])
        inter_missing_info[inter_name] = select_itersections(inter_info, ids)

    filename = figure_filename(figure_compare_filename)
    plot_image_compare_grid(inter_missing_info, filename=filename)


@ex.automain
@ex.capture
def main(models, plot):
    missings_info = {}

    for name, config in models.items():
        gpflow.reset_default_graph_and_session()

        model_type = config['type']
        filename = config['filename']

        test_data = get_test_dataset(model_type)
        x, y = test_data
        total_size = x.shape[0]

        print(f"Getting stats for '{name}' model...")

        if cache_file_exists(name):
            misses_results = cache_load_data(name)
        else:
            print(f"Loading '{name}' model...")
            model = load_model(filename, model_type)
            print("Compute misses...")
            misses_results = get_misses(model, test_data, name, config)

        Xm, Ym_true, Pm, probs, indices = misses_results
        missings_info[name] = misses_results

        if plot['missings']:
            print(f"Number of misses {Xm.shape[0]}")

            top_ns = list(range(1, num_classes()))
            tops = top_n_errors(top_ns, Pm, Ym_true, total_size)
            print(f"Top N errors {tops}")

            llm_misses = log_likelihood(Pm, Ym_true)
            print(f"LML misses {llm_misses}")

            llm_test = log_likelihood(probs, y)
            print(f"LML test {llm_test}")

            print("Plot missed images...")
            plot_missed_images(Xm[:100], Pm[:100], Ym_true[:100], filename=figure_filename(name))

    if plot['compare']:
        compare_models(missings_info)
