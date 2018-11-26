#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../run.py -mc convgp_creator -d svhn -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"