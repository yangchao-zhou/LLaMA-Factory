#!/bin/bash
# Script to set up environment and run LLaMA-Factory training with specified configurations

# Terminate any existing LLaMA-Factory processes
pkill -f "llamafactory"

# Set PYTHONPATH to current directory and LLaMA-Factory project directory
export PYTHONPATH=$(pwd)
export PYTHONPATH=/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory

# Configure CUDA devices for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

# Set temporary directory for the project
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp

# Configure NCCL network interface
export NCCL_SOCKET_IFNAME=eth1

# Change to the LLaMA-Factory project directory
cd /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/

# Activate the conda environment 'nemo'
source /root/miniconda3/bin/activate nemo

# Remove previous log file for the current role index, if it exists
rm -rf train-$MLP_ROLE_INDEX.log

# Launch training in the background using nohup, redirecting output to a log file
nohup bash -c "FORCE_TORCHRUN=1 NNODES=$MLP_WORKER_NUM NODE_RANK=$MLP_ROLE_INDEX MASTER_ADDR=$MLP_WORKER_0_HOST MASTER_PORT=$MLP_WORKER_0_PORT /root/miniconda3/envs/nemo/bin/llamafactory-cli train /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/examples/train_full/mistral_full_sft_ds.yaml" > train-$MLP_ROLE_INDEX.log 2>&1 &

# Monitor the log file in real-time
tail -f train-$MLP_ROLE_INDEX.log