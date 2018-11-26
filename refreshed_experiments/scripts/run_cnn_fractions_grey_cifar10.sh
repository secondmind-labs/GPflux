#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../run.py -mc basic_cnn_creator -d grey_cifar10_100epc  -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=1 python ../run.py -mc basic_cnn_creator -d grey_cifar10_5percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=2 python ../run.py -mc basic_cnn_creator -d grey_cifar10_10percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=3 python ../run.py -mc basic_cnn_creator -d grey_cifar10_25percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &