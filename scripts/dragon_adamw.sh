#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

# Usage: bash gpt2_adamw.sh [RUN_NAME] [NUM_GPUS=1]
RUN_NAME=${1}
NUM_GPUS=${2:-1}

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --run_name $RUN_NAME \
    --model dragon \
    --d_model 768 \
    --n_head 12 \
    --n_layer 7 \
    --expand_factor 2 \
    --optim spam \
    --batch_size 512 \
    --device_batch_size 32 \
    --learning_rate 1.8e-3 \
    --num_iterations 1000 \
    --warmup_iters 1000 \
    --warmdown_iters 150 \
    --weight_decay 0.1 \
    --rmsnorm False \
    --sequence_length 1024 \
    --vocab_size 50304 \
    --input_bin 'data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin' \
    --val_loss_every 125 \
    --val_tokens 10485760 \
    --save_every 0 \
    --log_wandb True