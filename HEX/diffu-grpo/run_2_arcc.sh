#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET=arcc
RUN_NAME=arcc_20250908_arcc_adapter_checkpoints
MODEL_PATH=''
NUM_ITER=12 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 6560 diffu_grpo_train.py \
    --config slurm_scripts/train_arcc.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 10