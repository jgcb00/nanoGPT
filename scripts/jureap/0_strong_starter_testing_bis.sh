#!/usr/bin/env bash

#SBATCH --account=jureap140
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --error=logs/eval7.err            # standard error file
#SBATCH --output=logs/eval7.out 

curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

module load GCCcore/.13.3.0
module load Python NVHPC
source venv/bin/activate

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
export MASTER_PORT=54123
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

srun torchrun_jsc ${DISTRIBUTED_ARGS[@]} main.py "$@"
