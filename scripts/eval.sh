#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE="1"
export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"

torchrun --nproc_per_node=${2} eval_ddp.py \
    --run_dir ${1} \
    --tasks hellaswag,swde,squadv2,squad_completion,fda,nq_open,drop,mmlu,triviaqa,arc_easy,arc_challenge,piqa,winogrande \
    --batch_size ${3}

