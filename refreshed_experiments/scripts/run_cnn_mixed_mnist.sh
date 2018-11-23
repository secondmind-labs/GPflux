#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc basic_cnn_creator -d mixed_mnist1 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc basic_cnn_creator -d mixed_mnist2 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc basic_cnn_creator -d mixed_mnist3 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc basic_cnn_creator -d mixed_mnist4 -c BasicCNNConfig -t KerasClassificationTrainer -p "$1" -r 10 &