#!/usr/bin/env bash

python ../main.py -mc convgp_creator -d grey_cifar10 -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
