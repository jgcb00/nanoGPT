#!/bin/bash

module load GCCcore/.13.3.0
module load Python NVHPC
source build/venv/bin/activate # created during the build phase of the JUBE script

# to make compatible with JUBE, remove the #SBATCH lines as well as the "srun"

GPUS_PER_NODE=4
NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54125
export NCCL_SOCKET_IFNAME=ib0 # Prevent NCCL not figuring out how to initialize.
export GLOO_SOCKET_IFNAME=ib0 # Prevent Gloo not being able to communicate.

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --rdzv_id $SLURM_JOB_ID
    --rdzv_endpoint "$MASTER_ADDR":"$MASTER_PORT"
    --rdzv_backend c10d
)

torchrun_jsc ${DISTRIBUTED_ARGS[@]} build/fetch/nanoGPT/main.py \
    --run_name dragon-L-adamw \
    --setup_only \
    --model dragon \
    --d_model 512 \
    --n_heads 16 \
    --n_kv_heads 8 \
    --n_layers 8 \
    --use_kv_sharing \
    --use_swa \
    --no-qk-norm \
    --attn_type diff \
    --lin_attn_type gdn \
    --expand_factor 2 \
    --layer-norm-scaling \
    --scalable_softmax \
    --optim adamw \
    --batch_size 256 \
    --device_batch_size 8 \
    --learning_rate 9.7e-4 \
    --num_iterations 400 \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --weight_decay 0.1 \
    --sequence_length 4736 \
    --vocab_size 50304 \
    --input_bin 'build/fetch/nanoGPT/data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'build/fetch/nanoGPT/data/fineweb10B/fineweb_val_*.bin' \
