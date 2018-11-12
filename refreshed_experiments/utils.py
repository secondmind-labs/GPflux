import numpy as np


class Configuration:

    @classmethod
    def summary(cls):
        summary_str = []
        for name, value in cls.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            summary_str.append('{}_{}\n'.format(name, str(value)))
        return ''.join(summary_str)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def group_results(results):
    return np.array([result.history['loss'] for result in results]), \
           np.array([result.history['val_loss'] for result in results])


def labels_onehot_to_int(labels):
    return labels.argmax(axis=-1)[..., None].astype(np.int32)


def reshape_to_2d(x):
    return x.reshape(x.shape[0], -1)


def calc_multiclass_error(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
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
