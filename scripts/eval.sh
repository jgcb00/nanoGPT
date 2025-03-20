#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE="1"
#export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"

#torch run here can cause issue due to the port that might be already in used by another process
python eval.py \
    --run_dir ${1} \
    --tasks hellaswag,swde,fda

#hellaswag,swde,fda,arc_easy,arc_challenge,piqa,winogrande,lambada,openbookqa,squadv2
