#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

export WANDB_MODE=offline

torchrun --nproc_per_node=1 main.py \
    --run_name test_exp14 \
    --no-use_gate_attn \
    --use_gate \
    --no-groupnorm_unique \
    --no-groupnorm_unique_independent \
    --rmsnorm_weights \
    --rope_to_nope \
    --slw_warmup_iters 0.6 \
    --rope_theta_local 163 \
    --model dragon \
    --d_model 512 \
    --n_heads 8 \
    --n_kv_heads 8 \
    --n_layers 4 \
    --use_kv_sharing \
    --use_swa \
    --qk-norm \
    --attn_type diff \
    --lin_attn_type gdn \
    --global_attn_repart middle \
    --expand_factor 2 \
    --layer-norm-scaling \
    --scalable_softmax \
    --optim adamw \
    --batch_size 8 \
    --device_batch_size 8 \
    --learning_rate 1.605e-3 \
    --num_iterations 66342 \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --weight_decay 0.1 \
    --sequence_length 5888 \
    --vocab_size 50304 \
    --input_bin '../nanoGPT/data/fineweb100B/fineweb_train_*.bin' \
    --input_val_bin '../nanoGPT/data/fineweb100B/fineweb_val_*.bin' \
    --val_loss_every 250 \
    --val_tokens 235520 \
    --save_every 10000 \
    --eval_benchmarks_tasks 'hellaswag,swde,fda' \
    --no-eval_benchmarks \
    --no-evalpg19 \
    --no-log_wandb
