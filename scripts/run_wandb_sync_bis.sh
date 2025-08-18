#!/bin/bash
module load gcc/12.2.0 python/3.11.7 cuda/12.2 cudnn cutensor/1.5.0.3--gcc--12.2.0 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
source /leonardo_work/BOOST_LCustodi/script/training/flex_fa_training_env/bin/activate
#module load GCC && module load Python/3.12.3 && module load NVHPC && module load cuDNN/9.5.0.50-CUDA-12
#source /p/project1/jureap140/jupiter_env/bin/activate
for d in wandb/offline-*; do
  wandb sync "$d"
done