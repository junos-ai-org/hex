#!/bin/bash

############################################################
GPU_IDS=(0)
MASTER_PORT=29707
############################################################
TASKS=('math') # 'gsm8k' 'math' 'arcc' 'truthfulqa'
GEN_LENGTHS=(256) # 128 256 512
DIFF_STEPS=(128) # half of the gen_length
MODEL_PATHS=("/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Instruct") # abs path of your dllm
OUTPUT_PATH="results/" # abs path of your folder to contain HEX json file 
CHECKPOINT_PATH=("nope") # "nope" means no LoRa Adaptor! Using foundation model in MODEL_PATHS
SEEDS=(42)
TEMPERATURES=(0)

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    for diffusion_step in "${DIFF_STEPS[@]}"; do
      for model in "${MODEL_PATHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          for temperature in "${TEMPERATURES[@]}"; do
            for checkpoint_path in "${CHECKPOINT_PATH[@]}"; do
              # Set batch size based on generation length
              if [ "$gen_length" -eq 512 ]; then
                batch_size=4
              else
                batch_size=8
              fi
              
              echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
  
              CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
              --nproc_per_node $NUM_GPUS \
              --master_port $MASTER_PORT \
              HEX.py \
              --dataset $task \
              --batch_size $batch_size \
              --gen_length $gen_length \
              --diffusion_steps $diffusion_step \
              --output_dir results/ \
              --model_path $model \
              --seed $seed \
              --temperature $temperature \
              --checkpoint_path $checkpoint_path
            done
          done
        done
      done
    done
  done
done
