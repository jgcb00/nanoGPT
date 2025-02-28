#!/bin/bash
# Get command line arguments
ARCHITECTURE=$1
ATTN_TYPE=$2
OPTIMIZER=$3
RUN_NAME="${ARCHITECTURE}_${ATTN_TYPE}_${OPTIMIZER}"

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1
source /leonardo_work/BOOST_LCustodi/script/training/flex_fa_training_env/bin/activate
export WANDB_MODE=offline

if [ "$ARCHITECTURE" == "gpt" ]; then
 d_model=768
 n_layer=12
 expand_factor=1
elif [ "$ARCHITECTURE" == "dragon" ]; then
 d_model=768
 n_layer=7
 expand_factor=2
fi

if [ "$OPTIMIZER" == "spam" ]; then
 learning_rate=1e-3
 warmup_iters=0
elif [ "$OPTIMIZER" == "muon" ]; then
 learning_rate=0.015
 warmup_iters=50
elif [ "$OPTIMIZER" == "adamw" ]; then
 learning_rate=1e-3
 warmup_iters=50
fi

echo "RUN_NAME: $RUN_NAME"
echo "ARCHITECTURE: $ARCHITECTURE"
echo "ATTN_TYPE: $ATTN_TYPE"
echo "OPTIMIZER: $OPTIMIZER"
echo "LEARNING_RATE: $learning_rate"
echo "WARMUP_ITERS: $warmup_iters"
echo "d_model: $d_model"
echo "n_layer: $n_layer"

torchrun --nproc_per_node=4 main.py \
 --run_name $RUN_NAME \
 --model $ARCHITECTURE \
 --d_model $d_model \
 --n_head 12 \
 --n_layer $n_layer \
 --expand_factor $expand_factor \
 --optim $OPTIMIZER \
 --batch_size 512 \
 --device_batch_size 32 \
 --learning_rate $learning_rate \
 --num_iterations 5000 \
 --warmup_iters $warmup_iters \
 --warmdown_iters 750 \
 --weight_decay 0.1 \
 --sequence_length 1024 \
 --vocab_size 50304 \
 --input_bin 'data/fineweb10B/fineweb_train_*.bin' \
 --input_val_bin 'data/fineweb10B/fineweb_val_*.bin' \
 --val_loss_every 125 \
 --val_tokens 10485760 \
 --save_every 0 \
 --log_wandb True \
 --attn_type $ATTN_TYPE