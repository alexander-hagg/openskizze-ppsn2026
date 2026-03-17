#!/bin/bash
#SBATCH --job-name=unet_opt_benchmark
#SBATCH --output=results/exp8_performance_benchmark/unet_opt_benchmark_%j.out
#SBATCH --error=results/exp8_performance_benchmark/unet_opt_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# U-Net Pipeline Optimization Benchmark
# Tests 4 optimization levels:
#   1. Baseline (current implementation)
#   2. GPU domain grid construction
#   3. GPU grids + Numba JIT
#   4. GPU grids + Numba JIT + Pinned memory

echo "========================================"
echo "U-Net Pipeline Optimization Benchmark"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup
cd $SLURM_SUBMIT_DIR
source ~/.bashrc
mamba activate openskizze_klam_qd

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
echo ""

# Create output directory
mkdir -p results/exp8_performance_benchmark

# Run benchmark
echo "Running benchmark..."
python experiments/exp8_performance_benchmark/benchmark_unet_optimizations.py

echo ""
echo "End time: $(date)"
echo "========================================"
