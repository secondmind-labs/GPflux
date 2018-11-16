#!/usr/bin/env bash

python ../main.py -mc convgp_creator -d svhn -c ConvGPConfig -t ClassificationGPTrainer -p "$1"
