#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Usage: bash gpt2_adamw.sh [RUN_NAME] [NUM_GPUS=1]
RUN_NAME=${1}
NUM_GPUS=${2:-1}

torchrun --nproc_per_node=$NUM_GPUS main.py \
    --run_name $RUN_NAME \
    --model dragon \
    --attn_type diff \
    --lin_attn_type mamba2 \
    --layer_norm_scaling \
    --use_swa \
    --d_model 1280 \
    --n_heads 20 \
    --n_layers 20 \
    --n_kv_heads 20 \
    --expand_factor 2 \
    --optim adamw \
    --batch_size 128 \
    --device_batch_size 8 \
    --learning_rate 3e-4 \
    --num_iterations 1000 \
    --warmup_iters 0 \
    --warmdown_iters 150 \
    --weight_decay 0.1 \
    --sequence_length 4096 \
    --vocab_size 50304 \
    --input_bin 'data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin' \
    --val_loss_every 125 \
    --val_tokens 10485760 \
    --save_every 0 \
    --log_wandb