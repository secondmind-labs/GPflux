#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../run.py -mc convgp_creator -d svhn_100epc -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=1 python ../run.py -mc convgp_creator -d svhn_5percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=2 python ../run.py -mc convgp_creator -d svhn_10percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &
CUDA_VISIBLE_DEVICES=3 python ../run.py -mc convgp_creator -d svhn_25percent -c TickConvGPConfig -t ClassificationGPTrainer -p "$1" &