#!/bin/bash
#SBATCH --job-name=exp7_single
#SBATCH --output=logs/exp7_single_%A_%a.out
#SBATCH --error=logs/exp7_single_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# Train single-scale U-Nets for all 3 parcel sizes
# Array indices: 0-2 map to 60m, 120m, 240m

set -e

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Parcel sizes to train (3 sizes: 60, 120, 240m)
PARCEL_SIZES=(60 120 240)
SIZE=${PARCEL_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Training single-scale U-Net for parcel size: ${SIZE}m"

# Adjust batch size for larger parcels (>69m need more memory)
if [ $SIZE -gt 69 ]; then
    BATCH_SIZE=16
else
    BATCH_SIZE=32
fi

echo "Using batch size: $BATCH_SIZE"

# Run training
python experiments/exp7_multiscale_unet/train_multiscale_comparison.py \
    --mode single \
    --parcel-sizes $SIZE \
    --data-dir /home/ahagg2s/openskizze-klam21-optimization/results/exp1_gp_training_data \
    --output-dir results/exp7_multiscale_unet \
    --max-epochs 200 \
    --batch-size $BATCH_SIZE \
    --seed 42

echo "Job completed at $(date)"
