#!/bin/bash
##SBATCH --nodes=8
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:4
##SBATCH --time=24:00:00
##SBATCH --error=logs/experiment20_longrun_LR411.err
##SBATCH --output=logs/experiment20_longrun_LR411.out
##SBATCH --account=BOOST_LCustodi
##SBATCH --partition=boost_usr_prod

# uncomment sbatch directives, srun, gpu_per_node to 4, val loss every to 250

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1

source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

export WANDB_MODE=offline

GPUS_PER_NODE=1
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

torchrun ${DISTRIBUTED_ARGS[@]} main.py \
    --run_name exp20long_Dragon-L-GDN-LR4.11_splus \
    --global_attn_repart middle \
    --no-input_norm \
    --no-full_lambdas \
    --eps_rmsnorm 1.0e-6 \
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
    --optim splus \
    --batch_size 128 \
    --device_batch_size 1 \
    --learning_rate 4.11 \
    --num_iterations 66342 \
    --warmup_iters 0.0045 \
    --warmdown_iters 0.15 \
    --weight_decay 0.1 \
    --sequence_length 5888 \
    --vocab_size 50304 \
    --input_bin '../nanoGPT/data/fineweb100B/fineweb_train_*.bin' \
    --input_val_bin '../nanoGPT/data/fineweb100B/fineweb_val_*.bin' \
    --val_loss_every 10 \
    --val_tokens 10174464 \
    --save_every 10000 \
    --eval_benchmarks_tasks 'hellaswag,swde,fda' \
    --eval_benchmarks \
    --no-evalpg19 \
    --log_wandb
