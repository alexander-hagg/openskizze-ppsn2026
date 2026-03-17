#!/bin/bash
#SBATCH --job-name=eval_hpo_cross
#SBATCH --output=logs/eval_hpo_cross_%j.out
#SBATCH --error=logs/eval_hpo_cross_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# HPO Cross-Domain Evaluation Job Script
# ============================================================================
# Evaluates HPO-trained models on all datasets (optimized, random, combined)
# to assess generalization performance.
#
# Usage:
#   sbatch hpc/submit_evaluate_hpo_cross_domain.sh
#   sbatch --mem=128G hpc/submit_evaluate_hpo_cross_domain.sh  # More memory
# ============================================================================

echo "======================================================================"
echo "HPO Cross-Domain Evaluation Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration (can be overridden by environment variables)
MODELS_DIR="${MODELS_DIR:-results/exp3_hpo/hyperparameterization}"
DATA_DIR="${DATA_DIR:-results/exp1_gp_training_data/training_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-results/exp3_hpo/hyperparameterization/cross_domain}"
TEST_FRACTION="${TEST_FRACTION:-0.15}"
SEED="${SEED:-42}"

echo "Configuration:"
echo "  Models dir:     $MODELS_DIR"
echo "  Data dir:       $DATA_DIR"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Test fraction:  $TEST_FRACTION"
echo "  Seed:           $SEED"
echo "  Using CPU only (--no-gpu flag)"
echo ""

# Check if models exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found: $MODELS_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

MODEL_COUNT=$(find "$MODELS_DIR" -name "model_*.pth" | wc -l)
echo "Found $MODEL_COUNT model files in $MODELS_DIR"
echo ""

# Run cross-domain evaluation
echo "======================================================================"
echo "Running Cross-Domain Evaluation"
echo "======================================================================"

python experiments/exp3_hpo/evaluate_hpo_cross_domain.py \
    --models-dir "$MODELS_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED" \
    --no-gpu

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Complete"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Cross-domain evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
