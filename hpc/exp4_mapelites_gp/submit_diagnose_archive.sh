#!/bin/bash
#SBATCH --job-name=diagnose_gp
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=any

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# Diagnose GP Extrapolation in MAP-Elites Archives
# ============================================================================
# Runs diagnostics to understand why MAP-Elites archives validate poorly:
# 1. Genome distribution shift analysis
# 2. GP uncertainty quantification
#
# Usage:
#   # Single archive
#   sbatch --export=ALL,NUM_EMITTERS=64,BATCH_SIZE=16,REPLICATE=1 \
#       hpc/exp4_mapelites_gp/submit_diagnose_archive.sh
#
#   # Batch all archives
#   bash hpc/exp4_mapelites_gp/submit_diagnose_all_archives.sh
# ============================================================================

# Get configuration from environment (or use defaults)
NUM_EMITTERS=${NUM_EMITTERS:-64}
BATCH_SIZE=${BATCH_SIZE:-16}
REPLICATE=${REPLICATE:-1}

echo "======================================================================"
echo "GP Extrapolation Diagnostics"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Change to project root
cd /home/ahagg2s/openskizze-klam21-optimization

# Configuration
RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
ARCHIVE_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/archive_final.pkl"
GP_MODEL="${GP_MODEL:-results/exp3_hpo/hyperparameterization/model_combined_ind1000_random_rep1.pth}"
TRAINING_DATA="${TRAINING_DATA:-results/exp1_gp_training_data/training_datasets/dataset_combined.npz}"
OUTPUT_DIR="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/diagnostics"

echo "Configuration:"
echo "  Archive:       $ARCHIVE_PATH"
echo "  GP Model:      $GP_MODEL"
echo "  Training Data: $TRAINING_DATA"
echo "  Output:        $OUTPUT_DIR"
echo ""

# Check if archive exists
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "ERROR: Archive not found: $ARCHIVE_PATH"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "ERROR: Training data not found: $TRAINING_DATA"
    exit 1
fi

# Check if GP model exists
if [ ! -f "$GP_MODEL" ]; then
    echo "ERROR: GP model not found: $GP_MODEL"
    exit 1
fi

# Check if diagnostics already completed
if [ -f "${OUTPUT_DIR}/diagnostic_metrics.yaml" ]; then
    echo "Diagnostics already complete for this archive."
    echo "To re-run, delete: ${OUTPUT_DIR}/diagnostic_metrics.yaml"
    exit 0
fi

# Run diagnostics
echo "======================================================================"
echo "Running Diagnostics"
echo "======================================================================"

python experiments/exp4_mapelites_gp/diagnose_gp_extrapolation.py \
    --archive "$ARCHIVE_PATH" \
    --gp-model "$GP_MODEL" \
    --training-data "$TRAINING_DATA" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Diagnostics completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - diagnostic1_genome_distribution_shift.png/pdf"
    echo "  - diagnostic2_gp_uncertainty.png/pdf"
    echo "  - diagnostic_metrics.yaml"
fi

exit $EXIT_CODE
