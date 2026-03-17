#!/bin/bash
#SBATCH --job-name=exp7_multi
#SBATCH --output=logs/exp7_multi_%j.out
#SBATCH --error=logs/exp7_multi_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Train multi-scale U-Net on mixed parcel sizes

set -e

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

echo "Training multi-scale U-Net"

# Run training (use smaller batch size for mixed large grids)
python experiments/exp7_multiscale_unet/train_multiscale_comparison.py \
    --mode multi \
    --parcel-sizes 60 120 240 \
    --data-dir /home/ahagg2s/openskizze-klam21-optimization/results/sail_data \
    --output-dir results/exp7_multiscale_unet \
    --max-epochs 200 \
    --batch-size 16 \
    --seed 42

echo "Job completed at $(date)"
