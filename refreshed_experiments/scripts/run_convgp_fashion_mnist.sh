#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc convgp_creator -d fashion_mnist -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
