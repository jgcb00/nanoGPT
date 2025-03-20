#!/bin/bash

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

#export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE="1"
#export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"

#torch run here can cause issue due to the port that might be already in used by another process
echo "arc_easy"
python eval.py \
    --run_dir ${1} \
    --tasks arc_easy

echo "arc_challenge"
python eval.py \
    --run_dir ${1} \
    --tasks arc_challenge

echo "piqa"
python eval.py \
    --run_dir ${1} \
    --tasks piqa

echo "winogrande"
python eval.py \
    --run_dir ${1} \
    --tasks winogrande

echo "lambada"
python eval.py \
    --run_dir ${1} \
    --tasks lambada

echo "openbookqa"
python eval.py \
    --run_dir ${1} \
    --tasks openbookqa

echo "squadv2"
python eval.py \
    --run_dir ${1} \
    --tasks squadv2

#hellaswag,swde,fda,arc_easy,arc_challenge,piqa,winogrande,lambada,openbookqa,squadv2
