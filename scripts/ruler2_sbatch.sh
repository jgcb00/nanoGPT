#!/bin/bash
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1         # number of gpus per node
#SBATCH --time=10:00:00              # time limits: here 1 hour
#SBATCH --error=logs/2eval.err            # standard error file
#SBATCH --output=logs/2eval.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_usr_prod # partition name for prod

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE="1"
#export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"

python eval.py \
    --run_dir logs/exp21_Dragon-L-GDN-gqa_gtda_adamw_9a399b02 \
    --tasks hellaswag,swde,fda \

python eval.py \
    --run_dir logs/exp21_Dragon-L-GDN-gqa_gtda_adamw_9a399b02 \
    --tasks niah_single_3 \
    --prompt_len 512,1024,2048,3074,4096,8192 \

# niah_single_3

# hellaswag,fda,swde
#hellaswag,swde,fda,arc_easy,arc_challenge,piqa,winogrande,lambada,openbookqa,squadv2
