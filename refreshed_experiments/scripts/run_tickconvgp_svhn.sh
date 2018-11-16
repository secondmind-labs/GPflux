#!/usr/bin/env bash

python ../main.py -mc convgp_creator -d svhn -c TickConvGPConfig -t ClassificationGPTrainer -p "$1"
