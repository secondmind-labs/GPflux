#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -mc convgp_creator -d grey_cifar10 -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
