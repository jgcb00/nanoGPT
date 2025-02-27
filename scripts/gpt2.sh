#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 main.py \
    --d_model 768 \
    --n_head 12 \
    --n_layer 12 \
    --optim muon \
    --batch_size 512 \
    --device_batch_size 16 \
    --num_iterations 4578 \
    --warmup_iters 0 \
    --warmdown_iters 1308 \
    --weight_decay 0 \
    --sequence_length 1024 \
    --vocab_size 50304 \
    --input_bin 'data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin' \
    --val_loss_every 125 \
    --val_tokens 10485760 \
    --save_every 0