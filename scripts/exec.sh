#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/flex_fa_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1

python test_mamba2_bis.py
#python test_inference.py
#python try.py
#python lm_eval_tests.py
#python test_decode.py