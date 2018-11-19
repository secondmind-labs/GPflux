#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc rbf_gp_creator -d fashion_mnist -c RBFGPConfig -t ClassificationGPTrainer -p "$1"
