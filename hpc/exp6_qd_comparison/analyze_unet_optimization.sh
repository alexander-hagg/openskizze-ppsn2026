#!/bin/bash
#SBATCH --job-name=unet_optim
#SBATCH --output=logs/unet_optimization_%j.out
#SBATCH --error=logs/unet_optimization_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# U-Net Optimization Analysis
# Tests FP16, torch.compile, smaller architectures

echo "========================================================================"
echo "U-NET OPTIMIZATION ANALYSIS"
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

# Run optimization analysis
python experiments/exp6_qd_comparison/analyze_unet_optimization.py \
    --batch-size 1024 \
    --num-warmup 5 \
    --num-runs 20 \
    --output results/exp6_qd_comparison/unet_optimization_analysis.json \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Analysis complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
