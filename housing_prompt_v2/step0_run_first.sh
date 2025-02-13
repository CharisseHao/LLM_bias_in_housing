#!/bin/bash
apt update
apt install jq python3-dev build-essential nvtop htop jnettop zip unzip nano less -y
pip install huggingface-hub
pip install vllm openpyxl pandas seaborn tqdm statsmodels
