#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ../main.py -e convgp_cifar10 --path "$1" &
CUDA_VISIBLE_DEVICES=1 python ../main.py -e convgp_cifar10_5percent --path "$1" &
CUDA_VISIBLE_DEVICES=2 python ../main.py -e convgp_cifar10_10percent --path "$1" &
CUDA_VISIBLE_DEVICES=3 python ../main.py -e convgp_cifar10_25percent --path "$1" &

