#!/bin/bash

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=48994
NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
echo "Master Address : "$MASTER_ADDR" | "$NUM_NODES" Nodes | World Size : "$WORLD_SIZE

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --rdzv_id $SLURM_JOB_ID
    --rdzv_endpoint $MASTER_ADDR:29500
    --rdzv_backend c10d
)

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --rdzv_id $SLURM_JOB_ID
    --rdzv_endpoint $MASTER_ADDR:29500
    --rdzv_backend c10d
)

# BASE MODEL FOR COMPARISON

# For 10B tokens model
# BS = 297459

srun torchrun ${DISTRIBUTED_ARGS[@]} main.py \
    --run_name dragon-L-adamw \
    --model dragon \
    --d_model 1280 \
    --n_heads 20 \
    --n_kv_heads 10 \
    --n_layers 20 \
    --use_kv_sharing \
    --use_swa \
    --no-qk-norm \
    --attn_type diff \
    --lin_attn_type gdn \
    --expand_factor 2 \
    --layer-norm-scaling \
    --scalable_softmax \
    --optim adamw \
    --batch_size 64 \
    --device_batch_size 2 \
    --learning_rate 9.7e-4 \
    --num_iterations 32990 \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --weight_decay 0.1 \
    --sequence_length 4736 \
    --vocab_size 50304 \
    --input_bin '/p/project1/jureap140/nanoGPT/data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin '/p/project1/jureap140/nanoGPT/data/fineweb10B/fineweb_val_*.bin' \
    --val_loss_every 250 \
    --val_tokens 10002432 \
    --save_every 10000 \
    --eval_benchmarks_tasks 'hellaswag,swde,fda' \
    --eval_benchmarks \
    --no-evalpg19 \
    --no-log_wandb