#!/bin/bash

sudo systemctl stop networking
sudo systemctl stop network-manager
sudo ip address add dev eth0 192.168.1.3/24

trap 'kill %1; kill %2' SIGINT
pdm run pipeline.py & pdm run src/sft_cv/net/inference_engine.py
wait
trap - SIGINT

echo "amo fatto"
