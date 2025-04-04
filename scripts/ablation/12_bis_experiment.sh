#!/bin/bash
#SBATCH --nodes=4           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4         # number of gpus per node
#SBATCH --time=24:00:00              # time limits: here 1 hour
#SBATCH --error=logs/experiment12bis_scoring_softrho1_a0.5.err            # standard error file
#SBATCH --output=logs/experiment12bis_scoring_softrho1_a0.5.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_usr_prod # partition name for prod

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1

source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

export WANDB_MODE=offline

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

# Dragon with :
# SWA
# +GQA
# +cross-layer KV sharing
# +layer-norm scaling
# +diff-attention


# For 10B tokens model
# BS = 297459

srun torchrun ${DISTRIBUTED_ARGS[@]} main.py \
    --run_name exp12_Dragon-L-GDN-diff-scores_softrho1_a0.5-adamw \
    --model dragon \
    --scoring_bin 'data/exp0_GPT2-xs-scorer_25f0c569/scores_*.bin' \
    --scores_loss_coupling 'soft-rho1' \
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
    --input_bin '../nanoGPT/data/fineweb100B/fineweb_train_*.bin' \
    --input_val_bin '../nanoGPT/data/fineweb100B/fineweb_val_*.bin' \
    --val_loss_every 250 \
    --val_tokens 10002432 \
    --save_every 10000 \
    --log_wandb \
    --no-eval_benchmarks \
    --no-evalpg19
