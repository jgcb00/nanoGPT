#!/bin/bash

# example of calling:
# srun run.sh small weak

# to make compatible with JUBE: 
# -remove the #SBATCH lines as well as the "srun"
# -change dir of virtual env, main.py and data files
# -remove logs/ creation, echos

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <model_size> [weak|strong]"
    echo "Available model sizes: starter, tiny, small, medium"
    echo "Second argument (optional):"
    echo "  weak   - Use --setup_only flag"
    echo "  strong - Use --no-setup_only flag (default)"
    exit 1
fi

MODEL_SIZE=$1
SETUP_MODE=${2:-strong}

if [ "$SETUP_MODE" = "weak" ]; then
    SETUP_FLAG="--setup_only"
else
    SETUP_FLAG="--no-setup_only"
fi

NUM_ITERS=10

case $MODEL_SIZE in
    starter)
        D_MODEL=512
        N_HEADS=8
        N_KV_HEADS=4
        N_LAYERS=8
        DEVICE_BATCH_SIZE=8
        BATCH_SIZE=512
        SEQUENCE_LENGTH=4736
        LOG_PREFIX="0stater"
        ;;
    tiny)
        D_MODEL=768
        N_HEADS=12
        N_KV_HEADS=6
        N_LAYERS=12
        DEVICE_BATCH_SIZE=2
        BATCH_SIZE=512
        SEQUENCE_LENGTH=18432
        LOG_PREFIX="1tiny"
        ;;
    small)
        D_MODEL=1024
        N_HEADS=16
        N_KV_HEADS=8
        N_LAYERS=12
        DEVICE_BATCH_SIZE=2
        BATCH_SIZE=2048
        SEQUENCE_LENGTH=14264
        LOG_PREFIX="2small"
        ;;
    medium)
        D_MODEL=1280
        N_HEADS=20
        N_KV_HEADS=10
        N_LAYERS=20
        DEVICE_BATCH_SIZE=2
        BATCH_SIZE=12288
        SEQUENCE_LENGTH=5248
        LOG_PREFIX="3medium"
        ;;
    *)
        echo "Invalid model size: $MODEL_SIZE"
        echo "Available model sizes: tiny, small, medium"
        exit 1
        ;;
esac

module load GCCcore/.13.3.0
module load Python NVHPC
source build/venv/bin/activate # created during the build phase of the JUBE script

GPUS_PER_NODE=4
NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    MASTER_ADDR="$MASTER_ADDR"i # allow communication over InfiniBand cells on JSC machines.
fi
export MASTER_PORT=54125
export NCCL_SOCKET_IFNAME=ib0 # prevent NCCL not figuring out how to initialize.
export GLOO_SOCKET_IFNAME=ib0 # prevent Gloo not being able to communicate.

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
    --run_name "dragon-${MODEL_SIZE}-adamw" \
    ${SETUP_FLAG} \
    --model dragon \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_kv_heads $N_KV_HEADS \
    --n_layers $N_LAYERS \
    --use_kv_sharing \
    --use_swa \
    --no-qk-norm \
    --attn_type diff \
    --lin_attn_type gdn \
    --expand_factor 2 \
    --layer-norm-scaling \
    --scalable_softmax \
    --optim adamw \
    --batch_size $BATCH_SIZE \
    --device_batch_size $DEVICE_BATCH_SIZE \
    --learning_rate 9.7e-4 \
    --num_iterations $NUM_ITERS \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --weight_decay 0.1 \
    --sequence_length $SEQUENCE_LENGTH \
    --vocab_size 50304 \
    --input_bin 'build/fetch/nanoGPT/data/fineweb10B/fineweb_train_*.bin' \
    --input_val_bin 'build/fetch/nanoGPT/data/fineweb10B/fineweb_val_*.bin' \
