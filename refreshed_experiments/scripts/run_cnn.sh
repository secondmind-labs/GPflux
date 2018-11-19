#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mnist -c BasicCNNConfig -t KerasClassificationTrainer -p "$1"
CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d fashion_mnist -c BasicCNNConfig -t KerasClassificationTrainer -p "$1"
CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d grey_cifar10 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1"
CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d svhn -c BasicCNNConfig -t KerasClassificationTrainer -p "$1"