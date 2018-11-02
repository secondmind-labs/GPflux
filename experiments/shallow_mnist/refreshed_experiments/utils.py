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
