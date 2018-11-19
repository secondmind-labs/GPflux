#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python ../main.py -mc rbf_gp_creator -d grey_cifar10 -c RBFGPConfig -t ClassificationGPTrainer -p "$1"
