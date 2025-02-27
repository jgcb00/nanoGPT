#!/bin/bash

torchrun --nproc_per_node=1 main.py \
    # arch - general
    --d_model 768 \
    --n_head 12 \
    --n_layer 12 \
    \
    # optim
    --optim muon \
    --batch_size 512 \
    --device_batch_size 64 \
    --num_iterations 4578 \
    --warmup_iters 0 \
    --warmdown_iters 1308 \
    --weight_decay 0 \
    \
    # data
    --sequence_length 1024 \
    --vocab_size 50304 \
    --input_bin 'data/my_dataset/train_*.bin' \
    --input_val_bin 'data/my_dataset/val_*.bin' \
    \
    # evaluation and logging
    --val_loss_every 125 \
    --val_tokens 10485760 \
    --save_every 0