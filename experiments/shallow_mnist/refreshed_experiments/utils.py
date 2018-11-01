import numpy as np


class Configuration:

    @classmethod
    def summary(cls):
        summary = ''
        for name, value in cls.__dict__.items():
            if name.startswith('_'):
                # discard protected and private members
                continue
            summary += '_'.join((name, str(value))) + '\n'
        return summary


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def group_histories(histories):
    validation_loss = []
    test_loss = []
    for h in histories:
        validation_loss.append(h.history['loss'])
        test_loss.append(h.history['val_loss'])
    return np.array(validation_loss), np.array(test_loss)