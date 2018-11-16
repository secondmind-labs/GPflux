#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python ../main.py -mc convgp_creator -d grey_cifar10 -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
