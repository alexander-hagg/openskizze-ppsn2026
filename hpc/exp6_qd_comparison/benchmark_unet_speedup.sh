#!/bin/bash
#SBATCH --job-name=unet_benchmark
#SBATCH --output=logs/unet_benchmark_%j.out
#SBATCH --error=logs/unet_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# U-Net QD Optimization Performance Benchmark
# Tests OLD vs NEW approach to show speedup from eliminating double phenotype expression

echo "========================================================================"
echo "U-NET QD OPTIMIZATION PERFORMANCE BENCHMARK"
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

# Run benchmark with realistic parameters
echo "Running benchmark..."
echo "  Batch size: 1024 (realistic: 128 emitters × 8 solutions per generation)"
echo "  Num runs: 20 (for statistical significance)"
echo "  Num warmup: 5 (ensure JIT compilation and GPU warmup)"
echo ""

python experiments/exp6_qd_comparison/benchmark_unet_speedup.py \
    --batch-size 1024 \
    --num-runs 20 \
    --num-warmup 5 \
    --unet-model results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth \
    --output results/exp6_qd_comparison/benchmark_results.json \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Benchmark complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
