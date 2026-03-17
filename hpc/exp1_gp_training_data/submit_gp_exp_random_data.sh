#!/bin/bash
#SBATCH --job-name=gp_exp_random
#SBATCH --output=logs/random_data_%A_%a.out
#SBATCH --error=logs/random_data_%A_%a.err
#SBATCH --array=0-2           # 3 parcel sizes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=72:00:00
#SBATCH --mem=350G
#SBATCH --partition=hpc

# ============================================================================
# GP EXPERIMENT: Phase 1B - Generate Random (Sobol) Training Data
# ============================================================================
#
# Generates random samples using scrambled Sobol sequences and evaluates
# them with KLAM_21 simulation.
#
# Usage:
#   mkdir -p logs
#   sbatch hpc/submit_gp_exp_random_data.sh
#
# Monitor:
#   squeue -u $USER
#   ls -lh results/random_data/*.npz | wc -l
#
# Expected outputs:
#   4 NPZ files: results/random_data/random_sobol_<size>m_n15400_seed*.npz
#
# NOTE: Each job generates 15,400 samples per size = 61,600 total random samples
# ============================================================================

echo "=========================================="
echo "GP EXPERIMENT: Random Data Generation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Parcel sizes (divisible by xy_scale=3)
PARCEL_SIZES=(60 120 240)

PARCEL_SIZE=${PARCEL_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Parcel Size: ${PARCEL_SIZE}m × ${PARCEL_SIZE}m"
echo "Samples: 15400"
echo "Start Time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p results/exp1_gp_training_data/random_data
mkdir -p logs

# Unique seed for each job (base_seed + task_id ensures no collisions)
# Random jobs use seeds 2000-2012 (13 unique seeds, separate from SAIL)
SEED=$((2000 + SLURM_ARRAY_TASK_ID))
echo "Random Seed: $SEED"

# Activate your conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Run random data generation
# Using 15,400 samples per size = 200,200 total (balanced with subsampled SAIL)
# Skip existing is ON by default - use --force to regenerate
# --collect-spatial-data saves full KLAM_21 inputs/outputs for each sample
python experiments/exp1_gp_training_data/generate_random_data.py \
    --parcel-size $PARCEL_SIZE \
    --num-samples 15400 \
    --num-workers 128 \
    --batch-size 128 \
    --output-dir results/exp1_gp_training_data/random_data \
    --seed $SEED \
    --scramble \
    --collect-spatial-data

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: ${PARCEL_SIZE}m random samples"
else
    echo "✗ FAILED: ${PARCEL_SIZE}m random samples"
fi

echo "=========================================="
exit $EXIT_CODE
