#!/bin/bash

set -e

# Script to run training with SMgNO model
# Usage: ./run_smgnonet.sh [NPROC_PER_NODE]
# Example: ./run_smgnonet.sh 1

NPROC_PER_NODE=${1:-1}

echo "Starting SMgNO training with $NPROC_PER_NODE processes..."

# Set environment variables
export PYTHONPATH="/home/yjlee/ace-main:$PYTHONPATH"
export WANDB_JOB_TYPE=training

# Run training with SMgNO config
torchrun --nproc_per_node $NPROC_PER_NODE -m fme.ace.train /home/yjlee/ace-main/train_output/config_smgnonet.yaml

echo "SMgNO training completed!"
