#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc convgp_creator -d mixed_mnist1 -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc convgp_creator -d mixed_mnist2 -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc convgp_creator -d mixed_mnist3 -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc convgp_creator -d mixed_mnist4 -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
