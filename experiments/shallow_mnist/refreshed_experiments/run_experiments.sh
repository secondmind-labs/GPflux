#!/usr/bin/env bash
python main.py \
-e cnn_mnist \
cnn_mnist_5percent \
cnn_mnist_10percent \
cnn_mnist_25percent \
cnn_cifar10 \
--path "$1"
