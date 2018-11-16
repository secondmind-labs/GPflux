#!/usr/bin/env bash

python ../main.py -mc convgp_creator -d fashion_mnist -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
