#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc rbf_gp_creator -d mnist -c RBFGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc rbf_gp_creator -d fashion_mnist -c RBFGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc rbf_gp_creator -d grey_cifar10 -c RBFGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc rbf_gp_creator -d svhn -c RBFGPConfig -t ClassificationGPTrainer -p "$1" &
