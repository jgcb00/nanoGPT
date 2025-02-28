#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Export des variables d'environnement
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline

ARCHITECTURE=${1}
echo "Architecture: $ARCHITECTURE"

# Définition des valeurs à explorer
ATTN_TYPES=("normal" "diff")
OPTIMIZERS=("adam" "spam" "muon")


# Store job IDs
JOB_IDS=()

# Exécuter les combinaisons
for ATTN in "${ATTN_TYPES[@]}"; do
  for OPTIM in "${OPTIMIZERS[@]}"; do
    RUN_NAME="${ARCHITECTURE}_${ATTN}_${OPTIM}"
    
    # Submit the job with explicit arguments and sbatch parameters
    JOB_ID=$(sbatch \
             --nodes=1 \
             --ntasks-per-node=1 \
             --cpus-per-task=32 \
             --gres=gpu:4 \
             --job-name="$RUN_NAME" \
             --output="logs/${RUN_NAME}_%j.out" \
             --error="logs/${RUN_NAME}_%j.err" \
             --time=04:00:00 \
             --partition=boost_usr_prod \
             --account=BOOST_LCustodi \
             scripts/run_training.sh "$ARCHITECTURE" "$ATTN" "$OPTIM")
    
    JOB_ID=$(echo "$JOB_ID" | awk '{print $NF}')
    echo "Submitted job $JOB_ID for $RUN_NAME"
    JOB_IDS+=($JOB_ID)
  done
done

# Submit the final job that waits for all previous jobs
DEPENDENCY_STRING=$(printf ":%s" "${JOB_IDS[@]}")
DEPENDENCY_STRING=${DEPENDENCY_STRING:1} # Remove leading ":"

FINAL_JOB_ID=$(sbatch \
               --dependency=afterok:$DEPENDENCY_STRING \
               --nodes=1 \
               --ntasks-per-node=1 \
               --cpus-per-task=2 \
               --job-name="wandb_sync" \
               --output="logs/wandb_sync_%j.out" \
               --error="logs/wandb_sync_%j.err" \
               --time=02:00:00 \
               --partition=lrd_all_serial \
               --account=BOOST_LCustodi \
               scripts/run_wandb_sync.sh
)

FINAL_JOB_ID=$(echo "$FINAL_JOB_ID" | awk '{print $NF}')
echo "Submitted final job $FINAL_JOB_ID to run after all training jobs finish."