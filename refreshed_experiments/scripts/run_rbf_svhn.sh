#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python ../main.py -mc rbf_gp_creator -d svhn -c RBFGPConfig -t ClassificationGPTrainer -p "$1"
