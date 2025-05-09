#!/bin/bash
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1         # number of gpus per node
#SBATCH --time=01:00:00              # time limits: here 1 hour
#SBATCH --error=logs/eval_13.err            # standard error file
#SBATCH --output=logs/eval_13.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_usr_prod # partition name for prod

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE="1"
#export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"

python eval_pg19.py \
    --run_dir logs/exp13_Dragon-L-scalable_softmax-dff-deepseekinit-adamw_98dc1763 \
    --num_samples 2048 \

python eval_pg19.py \
    --run_dir logs/exp13_Dragon-L-scalable_softmax-dff-deepseekinit-repart_middle-adamw_8192bc2a \
    --num_samples 2048 \

python eval_pg19.py \
    --run_dir logs/exp13_Dragon-L-GDN-scalable_softmax-dff-deepseekinit-skyladder_0.6-adamw_ad2556a5 \
    --num_samples 2048 \

python eval_pg19.py \
    --run_dir logs/exp12_Dragon-L-full_diff-adamw_16af55eb \
    --num_samples 2048 \

python eval_pg19.py \
    --run_dir logs/exp12_Dragon-L-diff_for_full-adamw_f0998ab9 \
    --num_samples 2048 \
