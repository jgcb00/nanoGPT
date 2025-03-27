GPUS_PER_NODE=1
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
)

# BASE MODEL FOR COMPARISON

# For 10B tokens model
# BS = 297459

torchrun ${DISTRIBUTED_ARGS[@]} main.py \
    --run_name exp1_GPT2-S-adamw \
    --d_model 768 \
    --n_heads 16 \
    --n_kv_heads 1 \
    --n_layers 12 \
    --optim adamw \
    --batch_size 512 \
    --device_batch_size 16 \
    --learning_rate 0.0036 \
    --num_iterations 2000 \
    --warmup_iters 0.045 \
    --warmdown_iters 0.20 \
    --weight_decay 0.1 \
    --sequence_length 1024 \
    --vocab_size 50304 \
    --input_bin 'data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'data/fineweb10B/fineweb_val_*.bin' \
    --val_loss_every 250 \
    --val_tokens 10485760 \
    --save_every 10000 \
    --log_wandb
