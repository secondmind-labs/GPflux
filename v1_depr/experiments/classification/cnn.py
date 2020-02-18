from tensorflow.keras import Sequential
from tensorflow.keras.initializers import constant, glorot_normal, truncated_normal
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.regularizers import l2


__all__ = 'cifar10_cnn_model_top cifar10_cnn_model'.split()


def cifar10_cnn_model_without_top():
    def dense(m, num):
        m.add(Dense(num,
              activation='relu',
              kernel_regularizer=l2(0.004),
              bias_initializer=constant(0.1),
              kernel_initializer=truncated_normal(stddev=0.04)))

    def conv(m, num, size, bias, input_shape=None):
        kwargs = dict(
            activation='relu',
            padding='same',
            bias_initializer=constant(bias),
            kernel_initializer=truncated_normal(stddev=5e-2))

        if input_shape is not None:
            kwargs['input_shape'] = input_shape

        m.add(Conv2D(num, (size, size), (1, 1), **kwargs))

    nn = Sequential()

    conv(nn, 64, 3, 0.0, input_shape=(32, 32, 3))
    nn.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    nn.add(BatchNormalization())

    conv(nn, 64, 5, 1.0)
    nn.add(BatchNormalization())
    nn.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    nn.add(Flatten())
    dense(nn, 384)
    dense(nn, 192)
    return nn


def cifar10_cnn_model_with_dropout():
    nn = cifar10_cnn_model_without_top()
    nn.add(Dropout(0.25))
    return nn


def cifar10_cnn_model():
    nn = cifar10_cnn_model_with_dropout()
    nn.add(Dense(10, activation='softmax', kernel_initializer=glorot_normal()))
    return nn