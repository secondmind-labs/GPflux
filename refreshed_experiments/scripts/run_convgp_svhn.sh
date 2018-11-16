#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python ../main.py -mc convgp_creator -d svhn -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
