#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../run.py -mc basic_cnn_creator -d mnist -c BasicCNNLongConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=1 python ../run.py -mc basic_cnn_creator -d fashion_mnist -c BasicCNNLongConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=2 python ../run.py -mc basic_cnn_creator -d grey_cifar10 -c BasicCNNLongConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=3 python ../run.py -mc basic_cnn_creator -d svhn -c BasicCNNLongConfig -t KerasClassificationTrainer -p "$1" -r 10 &