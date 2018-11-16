#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python ../main.py -mc convgp_creator -d fashion_mnist -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
