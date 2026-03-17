#!/bin/bash
#SBATCH --job-name=gp_exp_sail
#SBATCH --output=logs/sail_data_%A_%a.out
#SBATCH --error=logs/sail_data_%A_%a.err
#SBATCH --array=0-2          # 3 replicates for 240m only (MaxArraySize=5 on this cluster)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=72:00:00
#SBATCH --mem=350G
#SBATCH --partition=hpc

# ============================================================================
# GP EXPERIMENT: Phase 1A - Generate SAIL (Optimized) Training Data
# ============================================================================
#
# Runs SAIL optimization for each parcel size and replicate to generate
# optimized training data for the GP comparison experiment.
#
# NOTE: Cluster has MaxArraySize=5, so we can only run 5 array tasks at once.
#       Currently configured for 240m only (3 replicates).
#       To run other sizes, change PARCEL_SIZE below.
#
# Usage:
#   mkdir -p logs
#   sbatch hpc/submit_gp_exp_sail_data.sh
#
# Monitor:
#   squeue -u $USER
#   ls -lh results/sail_data/*.npz | wc -l
#
# Expected outputs:
#   12 NPZ files: results/sail_data/sail_<size>m_rep<N>_gen100000.npz
# ============================================================================

echo "=========================================="
echo "GP EXPERIMENT: SAIL Data Generation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Fixed parcel size (change this to run different sizes)
# Options: 60, 120, 240
PARCEL_SIZE=240

# Replicate from array task ID (0-2 -> 1-3)
REPLICATE=$((SLURM_ARRAY_TASK_ID + 1))

echo "Parcel Size: ${PARCEL_SIZE}m x ${PARCEL_SIZE}m"
echo "Replicate: ${REPLICATE}"
echo "Start Time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p results/exp1_gp_training_data/sail_data
mkdir -p logs

# Unique seed for each job (base_seed + task_id ensures no collisions)
# SAIL jobs use seeds 1000-1038 (39 unique seeds)
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
echo "Random Seed: $SEED"

# Activate your conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Run SAIL data generation
# Skip existing is ON by default - use --force to regenerate
# --collect-spatial-data saves full KLAM_21 inputs/outputs for each sample
python experiments/exp1_gp_training_data/generate_sail_data.py \
    --parcel-size $PARCEL_SIZE \
    --replicate $REPLICATE \
    --num-generations 100000 \
    --num-workers 128 \
    --output-dir results/exp1_gp_training_data/sail_data \
    --seed $SEED \
    --collect-spatial-data

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: ${PARCEL_SIZE}m rep${REPLICATE}"
else
    echo "✗ FAILED: ${PARCEL_SIZE}m rep${REPLICATE}"
fi

echo "=========================================="
exit $EXIT_CODE
