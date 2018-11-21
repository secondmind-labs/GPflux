#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc convgp_creator -d fashion_mnist_100epc -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc convgp_creator -d fashion_mnist_5percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc convgp_creator -d fashion_mnist_10percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc convgp_creator -d fashion_mnist_25percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
