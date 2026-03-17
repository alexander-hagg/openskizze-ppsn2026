#!/bin/bash
#SBATCH --job-name=unet_klam
#SBATCH --output=logs/unet_klam_%j.out
#SBATCH --error=logs/unet_klam_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# U-NET Training for KLAM_21 Spatial Prediction
# ============================================================================
# Trains a single U-NET model configuration.
# Configure via environment variables:
#   DATA_TYPE: "sail" or "random" (default: sail)
#   LOSS_TYPE: "mse" or "mse_grad" (default: mse)
#   SEED: Random seed (default: 42)
#   PARCEL_SIZE: Parcel size in meters (default: 60)
#
# Usage:
#   # Default: sail data, mse loss, seed 42
#   sbatch hpc/exp5_unet/submit_unet_training.sh
#
#   # Custom configuration:
#   sbatch --export=ALL,DATA_TYPE=random,LOSS_TYPE=mse_grad,SEED=43 hpc/exp5_unet/submit_unet_training.sh
# ============================================================================

echo "======================================================================"
echo "U-NET KLAM_21 Training Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create logs directory
mkdir -p logs
mkdir -p results/exp5_unet/unet_experiment

# Configuration (set via environment variables or use defaults)
DATA_TYPE="${DATA_TYPE:-sail}"
LOSS_TYPE="${LOSS_TYPE:-mse}"
SEED="${SEED:-42}"
PARCEL_SIZE="${PARCEL_SIZE:-60}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAIL_DATA_DIR="${SAIL_DATA_DIR:-results/exp1_gp_training_data/sail_data}"
RANDOM_DATA_DIR="${RANDOM_DATA_DIR:-results/exp1_gp_training_data/random_data}"

# Set data directory based on type
if [ "$DATA_TYPE" == "sail" ]; then
    DATA_DIR="$SAIL_DATA_DIR"
else
    DATA_DIR="$RANDOM_DATA_DIR"
fi

echo "Configuration:"
echo "  Data type:     $DATA_TYPE"
echo "  Data dir:      $DATA_DIR"
echo "  Loss type:     $LOSS_TYPE"
echo "  Seed:          $SEED"
echo "  Parcel size:   $PARCEL_SIZE"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo ""

# Output directory
OUTPUT_DIR="results/exp5_unet/unet_experiment/${DATA_TYPE}_${LOSS_TYPE}_seed${SEED}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
echo "======================================================================"
echo "Starting U-NET Training"
echo "======================================================================"

python experiments/exp5_unet/train_unet_klam.py \
    --data-type "$DATA_TYPE" \
    --data-dir "$DATA_DIR" \
    --parcel-size "$PARCEL_SIZE" \
    --loss "$LOSS_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Complete"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
else
    echo "Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
