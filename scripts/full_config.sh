#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/flex_fa_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

# Usage: bash full_config.sh [RUN_NAME] [NUM_GPUS=1]
RUN_NAME=${1}
NUM_GPUS=${2:-1}

# WARNING!! MUON+DRAGON = bad perfs

ARCH_ARGS=(
    --run_name $RUN_NAME
    --model gpt
    --d_model 768
    --n_heads 12
    --n_layers 12
    --expand_factor 1
    --attn_type normal
    --lin_attn_type mamba2
    --fused_loss_computation
)

ATTENTION_ARGS=(
    --n_kv_heads 12
    --no-use_kv_sharing
    --use_swa
    --swa_window_size 512
    --swa_warmup_iters 100
    --qk_norm
    --no-scalable_softmax
)

MAMBA_ARGS=(
    --no-rmsnorm
    --d_state 128
    --d_conv 4
    --headdim 64
    --ngroups 8
    --no-norm_before_gate
)

GDN_ARGS=(
    --use_gate
    --expand_v 2
)

OPTIM_ARGS=(
    --optim muon
    --batch_size 512
    --device_batch_size 32
    --num_iterations 1000
    --learning_rate 1e-4
    --warmup_iters 0
    --warmdown_iters 150
    --weight_decay 0.0
    --grad_norm_clip 1.0
)

DATA_ARGS=(
    --vocab_size 50304
    --sequence_length 1024
    --input_bin 'data/fineweb10B/fineweb_train_*.bin'
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin'
)

EVAL_ARGS=(
    --val_loss_every 125
    --val_tokens 10485760
    --save_every 0
    --log_wandb
)

torchrun --nproc_per_node=$NUM_GPUS main.py \
    "${ARCH_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    "${MAMBA_ARGS[@]}" \
    "${GDN_ARGS[@]}" \
    "${OPTIM_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
