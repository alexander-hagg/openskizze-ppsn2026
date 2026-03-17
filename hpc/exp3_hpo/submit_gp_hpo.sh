#!/bin/bash
#SBATCH --job-name=gp_hpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# GP HYPERPARAMETER OPTIMIZATION
# ============================================================================
#
# This script runs a systematic hyperparameter optimization for SVGP models.
#
# Hyperparameters explored:
#   - Number of inducing points: 100, 500, 1000, 2500, 5000
#   - K-means initialization: True, False
#
# Fixed settings:
#   - Max epochs: 200
#   - Early stopping patience: 20 epochs (validation loss)
#   - LR warmup: 50 epochs (linear)
#   - Batch size: 1024
#   - Learning rate: 0.01
#   - ARD Matern 2.5 kernel (62 lengthscales)
#
# Total configurations: 5 × 2 = 10 configs
# Total jobs: 10 configs × 3 datasets × 3 replicates = 90 jobs
#
# Prerequisites:
#   Run Phase 2 first to create training datasets:
#   python experiments/prepare_training_datasets.py
#
# Usage:
#   bash submit_all_gp_hpo.sh  # Submit all 90 jobs
#
#   Or submit individual jobs:
#   sbatch --export=ALL,DATASET=optimized,INDUCING=2000,KMEANS=1,REPLICATE=1 submit_gp_hpo.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/gp_hpo_*.out
#
# After completion, run analysis:
#   python experiments/exp3_hpo/analyze_hpo_results.py --results-dir results/exp3_hpo/hyperparameterization
# ============================================================================

# Get configuration from environment (or use defaults for testing)
DATASET=${DATASET:-optimized}
INDUCING=${INDUCING:-2000}
KMEANS=${KMEANS:-1}
REPLICATE=${REPLICATE:-1}

echo "=========================================="
echo "GP HYPERPARAMETER OPTIMIZATION"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Build config name
if [ "$KMEANS" -eq 1 ]; then
    KMEANS_FLAG="--kmeans-init"
    KMEANS_STR="kmeans"
else
    KMEANS_FLAG=""
    KMEANS_STR="random"
fi

CONFIG_NAME="ind${INDUCING}_${KMEANS_STR}"
RUN_NAME="${DATASET}_${CONFIG_NAME}_rep${REPLICATE}"

echo ""
echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Inducing points: ${INDUCING}"
echo "  K-means init: ${KMEANS_STR}"
echo "  Replicate: ${REPLICATE}"
echo "  Run name: ${RUN_NAME}"
echo ""
echo "Start Time: $(date)"
echo "=========================================="

# Create output directory
mkdir -p results/exp3_hpo/hyperparameterization
mkdir -p logs

# Activate conda environment
source /home/ahagg2s/miniforge3/bin/activate openskizze_klam_qd

# Check GPU
nvidia-smi || echo "No GPU found, using CPU"

# Run HPO training
python experiments/exp3_hpo/train_gp_hpo.py \
    --dataset $DATASET \
    --num-inducing $INDUCING \
    $KMEANS_FLAG \
    --replicate $REPLICATE \
    --data-dir results/exp1_gp_training_data/training_datasets \
    --output-dir results/exp3_hpo/hyperparameterization \
    --num-epochs 500 \
    --patience 20 \
    --warmup-epochs 50 \
    --batch-size 1024 \
    --learning-rate 0.01 \
    --test-fraction 0.15 \
    --val-fraction 0.15 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: ${RUN_NAME}"
else
    echo "✗ FAILED: ${RUN_NAME}"
fi

echo "=========================================="
exit $EXIT_CODE
