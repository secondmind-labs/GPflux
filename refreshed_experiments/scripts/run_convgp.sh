#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../run.py -mc convgp_creator -d mnist -c ConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=1 python ../run.py -mc convgp_creator -d fashion_mnist -c ConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=2 python ../run.py -mc convgp_creator -d grey_cifar10 -c ConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=3 python ../run.py -mc convgp_creator -d svhn -c ConvGPConfig -t ClassificationGPTrainer -p "$1" &
