#!/usr/bin/env bash
python ../main.py -e cnn_grey_cifar10 cnn_grey_cifar10_5percent \
cnn_grey_cifar10_10percent cnn_grey_cifar10_25percent -p "$1"
