#!/usr/bin/env bash

python ../main.py -mc convgp_creator -d fashion_mnist -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
