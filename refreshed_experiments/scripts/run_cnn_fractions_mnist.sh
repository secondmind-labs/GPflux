#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mnist_5percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10
CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mnist_10percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10
CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mnist_25percent -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10