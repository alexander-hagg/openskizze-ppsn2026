#!/bin/bash
#SBATCH --job-name=gp_exp_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --partition=gpu

# ============================================================================
# GP EXPERIMENT: Phase 3 - Train GP Models
# ============================================================================
#
# Trains SVGP models on each dataset type with multiple replicates.
#
# Prerequisites:
#   1. Run submit_gp_exp_sail_data.sh (Phase 1A)
#   2. Run submit_gp_exp_random_data.sh (Phase 1B)
#   3. Run prepare_training_datasets.py locally (Phase 2)
#
# Usage:
#   python experiments/prepare_training_datasets.py  # Run first!
#   bash submit_all_gp_training.sh  # Submit all 9 jobs
#
#   Or submit individual jobs:
#   sbatch --export=ALL,DATASET=optimized,REPLICATE=1 submit_gp_exp_training.sh
#
# Expected outputs:
#   9 model files: results/gp_experiment/gp_<dataset>_rep<N>.pth
# ============================================================================

# Get dataset and replicate from environment (or use defaults for testing)
DATASET=${DATASET:-optimized}
REPLICATE=${REPLICATE:-1}

echo "=========================================="
echo "GP EXPERIMENT: GP Training"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

echo "Dataset: ${DATASET}"
echo "Replicate: ${REPLICATE}"
echo "Start Time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p results/exp1_gp_training_data/gp_experiment
mkdir -p logs

# Activate your conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Check GPU
nvidia-smi || echo "No GPU found, using CPU"

# Run GP training
python experiments/exp1_gp_training_data/train_gp_experiment.py \
    --dataset $DATASET \
    --replicate $REPLICATE \
    --data-dir results/exp1_gp_training_data/training_datasets \
    --output-dir results/exp1_gp_training_data/gp_experiment \
    --num-inducing 2000 \
    --num-epochs 200 \
    --batch-size 1024 \
    --learning-rate 0.01 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: ${DATASET} rep${REPLICATE}"
else
    echo "✗ FAILED: ${DATASET} rep${REPLICATE}"
fi

echo "=========================================="
exit $EXIT_CODE
