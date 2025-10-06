#!/bin/bash

############################################################
GPU_IDS=(0)
MASTER_PORT=29704
############################################################
TASKS=('gsm8k' 'math' 'arcc' 'truthfulqa') 
GEN_LENGTHS=(256)
DIFF_STEPS=(128)
MODEL_PATHS=("") 
# CHECKPOINT_PATH=("") # LoRA adapter path
CHECKPOINT_PATH=("nope") # "nope" means no LoRa Adaptor! Using foundation model
DECODINGS=("random") # "low_confidence" (low_confidence means topk) "topk_margin" "random"
KV_CACHINGS=("baseline") # "baseline" "prefix_cache" "parallel" "parallel_factor" "prefix_cache_parallel" "dual_cache_parallel" "prefix_cache_parallel_factor" "dual_cache_parallel_factor" 
BLOCK_LENGTHS=(32)
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
        for decoding in "${DECODINGS[@]}"; do
          for caching in "${KV_CACHINGS[@]}"; do
            for block_length in "${BLOCK_LENGTHS[@]}"; do
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
                    eval.py \
                    --dataset $task \
                    --batch_size $batch_size \
                    --gen_length $gen_length \
                    --diffusion_steps $diffusion_step \
                    --decoding $decoding \
                    --output_dir "" \
                    --model_path $model \
                    --seed $seed \
                    --block_length $block_length \
                    --kv_caching $caching \
                    --temperature $temperature \
                    --checkpoint_path $checkpoint_path
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done


