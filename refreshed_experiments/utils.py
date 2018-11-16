import gpflow
import keras
import numpy as np

from gpflux.convolution import PatchHandler, ImagePatchConfig


def get_from_module(name, module):
    if hasattr(module, name):
        return getattr(module, name)
    else:
        available = ' '.join([item for item in dir(module) if not item.startswith('__')])
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


def get_dataset_fraction(dataset, fraction):
    (train_features, train_targets), (test_features, test_targets) = dataset.load_data()
    seed = np.random.get_state()
    # fix the seed for numpy, so we always get the same fraction of examples
    np.random.seed(0)
    train_ind = np.random.permutation(range(train_features.shape[0]))[
                :int(train_features.shape[0] * fraction)]
    train_features, train_targets = train_features[train_ind], train_targets[train_ind]
    np.random.set_state(seed)
    return (train_features, train_targets), (test_features, test_targets)


def get_dataset_fixed_examples_per_class(dataset, num_examples):
    (train_features, train_targets), (test_features, test_targets) = dataset.load_data()
    selected_examples = []
    selected_targets = []
    num_classes = set(train_targets)
    for i in num_classes:
        indices = train_targets == i
        selected_examples.append(train_features[indices][:num_examples])
        selected_targets.append(train_targets[indices][:num_examples])
    selected_examples = np.vstack(selected_examples)
    selected_targets = np.hstack(selected_targets)
    return (selected_examples, selected_targets), (test_features, test_targets)


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
