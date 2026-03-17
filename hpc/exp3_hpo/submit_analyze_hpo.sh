#!/bin/bash
#SBATCH --job-name=analyze_hpo
#SBATCH --output=logs/analyze_hpo_%j.out
#SBATCH --error=logs/analyze_hpo_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# HPO Analysis Job Script
# ============================================================================
# Analyzes hyperparameter optimization results including:
# - Summary statistics and visualizations
# - Statistical tests
# - Inference timing analysis (GPU accelerated)
#
# Usage:
#   sbatch hpc/submit_analyze_hpo.sh
#   sbatch --partition=gpu-v100 hpc/submit_analyze_hpo.sh  # For specific GPU
# ============================================================================

echo "======================================================================"
echo "HPO Analysis Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
RESULTS_DIR="${RESULTS_DIR:-results/exp3_hpo/hyperparameterization}"
OUTPUT_DIR="${OUTPUT_DIR:-results/exp3_hpo/hyperparameterization/analysis}"

echo "Configuration:"
echo "  Results dir: $RESULTS_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# Check GPU availability
nvidia-smi || echo "No GPU found, using CPU"

# Run analysis
echo "======================================================================"
echo "Running HPO Analysis"
echo "======================================================================"

python experiments/exp3_hpo/analyze_hpo_results.py \
    --results-dir "$RESULTS_DIR" \
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
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
else
    echo "Analysis failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
