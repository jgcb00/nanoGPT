#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/flex_fa_training_env/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

# Usage: bash full_config.sh [RUN_NAME] [NUM_GPUS=1]
RUN_NAME=${1}
NUM_GPUS=${2:-1}

ARCH_ARGS=(
    --run_name $RUN_NAME
    --model gated-delta-net
    --d_model 1280
    --n_heads 20
    --n_layers 36
    --expand_factor 1
    --attn_type normal
    --lin_attn_type mamba2
    --layer_norm_scaling
    --fused_loss_computation
)

ATTENTION_ARGS=(
    --n_kv_heads 20
    --no-use_kv_sharing
    --no-use_swa
    --swa_window_size 512
    --swa_warmup_iters 100
    --qk_norm
    --no-scalable_softmax
    --disable_scalable_softmax_for_local
    --no-rope_to_nope
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
    --batch_size 64
    --device_batch_size 2
    --num_iterations 1000
    --learning_rate 1e-4
    --warmup_iters 0
    --warmdown_iters 150
    --weight_decay 0.0
    --grad_norm_clip 1.0
    --scheduler wsd
)

DATA_ARGS=(
    --vocab_size 50304
    --sequence_length 4096
    --no-use_patch_level_training
    --patch_size 4
    --patch_training_fraction 0.67
    --input_bin 'data/fineweb10B/fineweb_train_*.bin'
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin'
)

EVAL_ARGS=(
    --val_loss_every 125
    --val_tokens 10485760
    --save_every 0
    --no-log_wandb
    --num_params 0
    --vocab_size_real 50257
    --no-eval_benchmarks
    --eval_benchmarks_tasks 'hellaswag,swde,fda,openbookqa,arc_easy,arc_challenge,piqa,winogrande,lambada,squadv2'
    --eval_tokenizer_path 'data/enc.pkl'
    --no-evalpg19
    --evalpg19_ctx_len 16384
    --evalpg19_num_samples 128
    --evalpg19_batch_size 4
)

torchrun --nproc_per_node=$NUM_GPUS main.py \
    "${ARCH_ARGS[@]}" \
    "${ATTENTION_ARGS[@]}" \
    "${MAMBA_ARGS[@]}" \
    "${GDN_ARGS[@]}" \
    "${OPTIM_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_ARGS[@]}"
