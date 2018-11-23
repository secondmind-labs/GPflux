#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mnist -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc basic_cnn_creator -d fashion_mnist -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc basic_cnn_creator -d grey_cifar10 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc basic_cnn_creator -d svhn -c CNNShortConfig -t KerasClassificationTrainer -p "$1" -r 10 &