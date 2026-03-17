#!/bin/bash
#SBATCH --job-name=benchmark_unet
#SBATCH --output=logs/benchmark_unet_%j.out
#SBATCH --error=logs/benchmark_unet_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# OpenSKIZZE - U-Net Throughput Benchmarking
# Measures inference throughput for real-time QD optimization

set -e

echo "=================================================="
echo "U-Net Inference Throughput Benchmark"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load cuda/default

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Model path
MODEL_DIR="results/exp5_unet/unet_experiment/sail_mse_seed42"
MODEL_PATH="${MODEL_DIR}/best_model.pth"
NORM_PATH="${MODEL_DIR}/normalization.json"

echo "Model: $MODEL_PATH"
echo "Normalization: $NORM_PATH"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$NORM_PATH" ]; then
    echo "ERROR: Normalization file not found at $NORM_PATH"
    exit 1
fi

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run benchmark
echo "Starting benchmark..."
echo ""

python experiments/exp5_unet/benchmark_unet_throughput.py \
    --model-path "$MODEL_PATH" \
    --norm-path "$NORM_PATH" \
    --device cuda \
    --batch-sizes 1 4 8 16 32 64 128 256 \
    --num-warmup 100 \
    --num-samples 2000 \
    --output "${MODEL_DIR}/throughput_benchmark.json"

echo ""
echo "=================================================="
echo "Benchmark completed!"
echo "End time: $(date)"
echo "Results saved to: ${MODEL_DIR}/throughput_benchmark.json"
echo "=================================================="
