#!/bin/bash
#SBATCH --job-name=mapelites_gp_sweep
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=hpc
#OLD --gres=gpu:1

# OpenSKIZZE KLAM-21 Optimization
# Copyright (C) 2025 [Alexander Hagg]
# Licensed under AGPLv3

# ============================================================================
# MAP-Elites GP Surrogate Grid Sweep
# ============================================================================
# Runs MAP-Elites with pre-trained GP model across different configurations
# of emitters and batch sizes.
#
# Grid: 5 emitter counts × 5 batch sizes × 3 replicates = 75 jobs
#   - num_emitters: [4, 16, 64, 256, 512]
#   - batch_size: [4, 16, 64, 256, 512]
#   - replicates: [1, 2, 3]
#
# Usage:
#   bash submit_all_mapelites_gp_sweep.sh  # Submit all 75 jobs
#
#   Or submit individual jobs:
#   sbatch --export=ALL,NUM_EMITTERS=64,BATCH_SIZE=16,REPLICATE=1 \
#       hpc/exp4_mapelites_gp/submit_mapelites_gp_sweep.sh
# ============================================================================

# Get configuration from environment (or use defaults for testing)
NUM_EMITTERS=${NUM_EMITTERS:-64}
BATCH_SIZE=${BATCH_SIZE:-16}
REPLICATE=${REPLICATE:-1}
SEED=$((99 + REPLICATE - 1))

echo "======================================================================"
echo "MAP-Elites GP Sweep Job"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Change to project root
cd /home/ahagg2s/openskizze-klam21-optimization

# Create logs and results directories
mkdir -p logs
mkdir -p results/exp4_mapelites_gp/mapelites_gp

echo "Configuration:"
echo "  GP Model:       $GP_MODEL"
echo "  Num Emitters:   $NUM_EMITTERS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Replicate:      $REPLICATE"
echo "  Seed:           $SEED"
echo "  Generations:    $NUM_GENERATIONS"
echo "  Parcel Size:    $PARCEL_SIZE"
echo "  Lambda UCB:     $LAMBDA_UCB"
echo "  Genome Bounds:  [-$GENOME_BOUNDS, $GENOME_BOUNDS]"
echo ""

# Check if GP model exists
if [ ! -f "$GP_MODEL" ]; then
    echo "ERROR: GP model not found: $GP_MODEL"
    exit 1
fi

# Output directory
OUTPUT_DIR="results/exp4_mapelites_gp/mapelites_gp/emit${NUM_EMITTERS}_batch${BATCH_SIZE}_rep${REPLICATE}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run experiment
echo "======================================================================"
echo "Running MAP-Elites"
echo "======================================================================"

python experiments/exp4_mapelites_gp/run_mapelites_gp.py \
    --gp-model "$GP_MODEL" \
    --num-emitters "$NUM_EMITTERS" \
    --batch-size "$BATCH_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --parcel-size "$PARCEL_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --lambda-ucb "$LAMBDA_UCB" \
    --genome-bounds "$GENOME_BOUNDS"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Complete"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
else
    echo "Experiment failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
