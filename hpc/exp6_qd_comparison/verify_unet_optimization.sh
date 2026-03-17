#!/bin/bash
#SBATCH --job-name=verify_optim
#SBATCH --output=logs/verify_optimization_%j.out
#SBATCH --error=logs/verify_optimization_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# Verify U-Net optimization speedup in actual evaluator pipeline

echo "========================================================================"
echo "VERIFY U-NET OPTIMIZATION SPEEDUP"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load conda
source ~/.bashrc
conda activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Project root
cd /home/ahagg2s/openskizze-klam21-optimization

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Run verification
python experiments/exp6_qd_comparison/verify_unet_optimization.py \
    --batch-size 1024 \
    --num-warmup 5 \
    --num-runs 20 \
    --unet-model results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth \
    --output results/exp6_qd_comparison/verify_optimization.json

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Verification complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
