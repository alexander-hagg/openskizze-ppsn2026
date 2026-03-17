#!/bin/bash
#SBATCH --job-name=unet_profile
#SBATCH --output=logs/unet_profile_%j.out
#SBATCH --error=logs/unet_profile_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# U-Net QD Evaluation - Detailed Profiling
# Breaks down where time is spent to identify actual bottlenecks

echo "========================================================================"
echo "U-NET QD EVALUATION - DETAILED PROFILING"
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
cd /home/alex/Documents/_cloud/Funded_Projects/OpenSKIZZE/code/openskizze-klam21-optimization

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Run detailed profiling
echo "Running detailed profiling..."
echo ""

python experiments/exp6_qd_comparison/profile_unet_bottlenecks.py \
    --batch-size 1024 \
    --num-runs 20 \
    --unet-model results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth \
    --output results/exp6_qd_comparison/profiling_results.json \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Profiling complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
