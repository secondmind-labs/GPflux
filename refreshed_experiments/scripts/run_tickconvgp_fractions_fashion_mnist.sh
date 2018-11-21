#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc convgp_creator -d mnist_100epc -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
CUDA_VISIBLE_DEVICES=1 python ../main.py -mc convgp_creator -d mnist_5percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
CUDA_VISIBLE_DEVICES=2 python ../main.py -mc convgp_creator -d mnist_10percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
CUDA_VISIBLE_DEVICES=3 python ../main.py -mc convgp_creator -d mnist_25percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
