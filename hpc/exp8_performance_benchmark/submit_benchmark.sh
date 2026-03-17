#!/bin/bash
#SBATCH --job-name=exp8_bench
#SBATCH --output=logs/exp8_benchmark_%j.out
#SBATCH --error=logs/exp8_benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Experiment 8: Performance Benchmarking
# Measures bottlenecks and validates optimizations

set -e

# Use SLURM_SUBMIT_DIR which is set to where sbatch was called from
PROJECT_ROOT="$SLURM_SUBMIT_DIR"

echo "=============================================="
echo "EXPERIMENT 8: PERFORMANCE BENCHMARKING"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "Project root: $PROJECT_ROOT"
echo "Started: $(date)"
echo ""

# Setup environment
cd "$PROJECT_ROOT"
source ~/.bashrc
conda activate openskizze_klam_qd

# Create output directories
mkdir -p results/exp8_performance_benchmark

# Run benchmarks
echo "Running all benchmarks..."
python experiments/exp8_performance_benchmark/run_benchmark.py \
    --benchmark all \
    --batch-sizes 8 16 32 64 128 256 512 1024 \
    --num-iterations 10 \
    --parcel-size 60 \
    --output-dir results/exp8_performance_benchmark \
    --device cuda

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo "Finished: $(date)"
echo "Results: results/exp8_performance_benchmark/benchmark_results.json"
