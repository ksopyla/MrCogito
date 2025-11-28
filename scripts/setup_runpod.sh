#!/bin/bash

cp /workspace/ssh/id* $HOME/.ssh/

export POETRY_HOME="/workspace/poetry/"
export POETRY_CACHE_DIR=/workspace/poetry/cache 
export HF_HOME=/workspace/hf_home/
export HF_DATASETS_CACHE=/workspace/hf_home/datasets/


echo 'export HF_HOME="/workspace/hf_home/"' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE="/workspace/hf_home/datasets/"' >> ~/.bashrc
echo 'export POETRY_HOME="/workspace/poetry/"' >> ~/.bashrc
echo 'export POETRY_CACHE_DIR="/workspace/poetry/cache/"' >> ~/.bashrc

echo 'export PATH="/workspace/poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc