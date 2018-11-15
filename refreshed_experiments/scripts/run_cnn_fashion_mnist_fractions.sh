#!/usr/bin/env bash
python ../main.py -e cnn_fashion_mnist cnn_fashion_mnist_5percent cnn_fashion_mnist_10percent cnn_fashion_mnist_25percent -p "$1"
