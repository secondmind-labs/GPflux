#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py -e convgp_mnist --path "$1" &
CUDA_VISIBLE_DEVICES=1 python main.py -e convgp_mnist5percent --path "$1" &
CUDA_VISIBLE_DEVICES=2 python main.py -e convgp_mnist10percent --path "$1" &
CUDA_VISIBLE_DEVICES=3 python main.py -e convgp_mnist25percent --path "$1" &

