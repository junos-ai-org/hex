#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET="sudoku"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=''
NUM_ITER=12 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 1
