#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=450G
#SBATCH --job-name=w1280
#SBATCH --error=logs/exp14u_w1280_%j.err
#SBATCH --output=logs/exp14u_w1280_%j.out
#SBATCH --account=BOOST_LCustodi
#SBATCH --partition=boost_usr_prod
##SBATCH --account=jureap140
##SBATCH --partition=jureap
##SBATCH --nodelist=jpbo-009-[01-48]

# uncomment sbatch directives, distributed args, srun, number of gpus per node

module load gcc/12.2.0 python/3.11.7 cuda/12.2 cudnn cutensor/1.5.0.3--gcc--12.2.0 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#module load GCC && module load Python/3.12.3 && module load NVHPC && module load cuDNN/9.5.0.50-CUDA-12
#source /p/project1/jureap140/jupiter_env/bin/activate
#export TRITON_HOME="/p/project1/jureap140/temp"
#export WANDB_CACHE_DIR="/p/project1/jureap140/temp"
#export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_TIMEOUT=1200 # seconds

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

# d_model=512, n_heads=8, n_kv_heads=4, device_bs=8
# d_model=1024, n_heads=16, n_kv_heads=8, device_bs=4
# d_model=2048, n_heads=32, n_kv_heads=16, device_bs=2

srun torchrun ${DISTRIBUTED_ARGS[@]} main.py \
    --run_name test_uscaling_completep_w1280_LRh1_LRs2p-6_LRe2p-4_LRhead2p-6_noWDhead \
    --no-fused_loss_computation \
    --use_uscaling \
    --uscaling_tau 0.2 \
    --uscaling_dt_mul 1.0 \
    --init_std 1. \
    --softcap_global_attn 50.0 \
    --no-input_norm \
    --no-full_lambdas \
    --eps_rmsnorm 1.0e-6 \
    --groupnorm \
    --groupnorm_unique \
    --groupnorm_unique_independent \
    --rmsnorm_weights \
    --rope_to_nope \
    --slw_warmup_iters 0.6 \
    --rope_theta_local 163 \
    --model dragon \
    --d_model 1280 \
    --n_heads 20 \
    --n_kv_heads 10 \
    --n_layers 20 \
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
    --batch_size 64 \
    --device_batch_size 2 \
    --learning_rate 1.0 \
    --weight_decay 1e-4 \
    --uscaling_lr_scalar 1.56e-2 \
    --uscaling_lr_embed 6.25e-2 \
    --uscaling_lr_head 1.56e-2 \
    --num_iterations 32990 \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --sequence_length 4736 \
    --vocab_size 50304 \
    --input_bin '../../nanoGPT/data/fineweb100B/fineweb_train_*.bin' \
    --input_val_bin '../../nanoGPT/data/fineweb100B/fineweb_val_*.bin' \
    --val_loss_every 250 \
    --val_tokens 10002432 \
    --inspect_every 500 \
    --save_every 1000 \
    --eval_benchmarks_tasks 'hellaswag,swde,fda' \
    --eval_benchmarks \
    --no-evalpg19 \
    --log_wandb

# 2p-4 : 6.25e-2
# 2p-6 : 1.56e-2
# 2p-8 : 3.91e-3