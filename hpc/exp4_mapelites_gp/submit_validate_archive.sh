#!/bin/bash
#SBATCH --job-name=validate_archive
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=any

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# Validate MAP-Elites Archives with Real KLAM_21
# ============================================================================
# Re-evaluates top solutions from MAP-Elites archives with the real KLAM_21
# physics simulation to validate GP predictions.
#
# This runs for each of the 75 sweep configurations (5 emitters × 5 batch × 3 rep)
#
# Usage:
#   bash submit_all_validate_archives.sh  # Submit all 75 jobs
#
#   Or submit individual jobs:
#   sbatch --export=ALL,NUM_EMITTERS=64,BATCH_SIZE=16,REPLICATE=1 \
#       hpc/exp4_mapelites_gp/submit_validate_archive.sh
#
# To validate a specific archive manually:
#   python experiments/exp4_mapelites_gp/validate_mapelites_archive.py \
#       --archive results/mapelites_gp/emit64_batch16_rep1/archive_final.pkl \
#       --top-n 50 --strategy top
# ============================================================================

# Get configuration from environment (or use defaults for testing)
NUM_EMITTERS=${NUM_EMITTERS:-64}
BATCH_SIZE=${BATCH_SIZE:-16}
REPLICATE=${REPLICATE:-1}

echo "======================================================================"
echo "MAP-Elites Archive Validation Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Change to project root
cd /home/ahagg2s/openskizze-klam21-optimization

# Create logs directory
mkdir -p logs

# Configuration
TOP_N="${TOP_N:-50}"           # Number of top solutions to validate
STRATEGY="${STRATEGY:-top}"    # Selection strategy: top, random, diverse

# Construct archive path
RUN_NAME="emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"
ARCHIVE_PATH="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/archive_final.pkl"

echo "Configuration:"
echo "  Archive:        $ARCHIVE_PATH"
echo "  Top N:          $TOP_N"
echo "  Strategy:       $STRATEGY"
echo "  Num Emitters:   $NUM_EMITTERS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Replicate:      $REPLICATE"
echo ""

# Check if archive exists
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "ERROR: Archive not found: $ARCHIVE_PATH"
    echo "Skipping this configuration."
    exit 0
fi

# Check if validation already completed
OUTPUT_DIR="results/exp4_mapelites_gp/mapelites_gp/${RUN_NAME}/validation_${STRATEGY}"
if [ -f "${OUTPUT_DIR}/validation_metrics.yaml" ]; then
    echo "Validation (${STRATEGY}) already complete for this archive."
    echo "To re-run, delete: ${OUTPUT_DIR}/validation_metrics.yaml"
    exit 0
fi

# Run validation
echo "Starting KLAM_21 validation (${STRATEGY})..."
echo "======================================================================"

python experiments/exp4_mapelites_gp/validate_mapelites_archive.py \
    --archive "$ARCHIVE_PATH" \
    --top-n "$TOP_N" \
    --strategy "$STRATEGY" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Validation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
