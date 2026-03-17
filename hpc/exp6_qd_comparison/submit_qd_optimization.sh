#!/bin/bash
#SBATCH --job-name=exp6_qd
#SBATCH --output=logs/exp6_qd_%j.out
#SBATCH --error=logs/exp6_qd_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu4
#SBATCH --gres=gpu:1

# Experiment 6: MAP-Elites with Offline Surrogates (single-job, env-var based)
#
# Required environment variables (pass via sbatch --export=ALL,...):
#   MODEL       - unet, svgp, or hybrid
#   UCB_LAMBDA  - UCB exploration weight (e.g. 0.0, 0.1, 1.0, 10.0)
#   SEED        - random seed (e.g. 42, 43, 44)
#
# Example:
#   sbatch --export=ALL,MODEL=unet,UCB_LAMBDA=0.0,SEED=42 hpc/exp6_qd_comparison/submit_qd_optimization.sh

set -e

# Validate required env vars
if [ -z "$MODEL" ] || [ -z "$UCB_LAMBDA" ] || [ -z "$SEED" ]; then
    echo "ERROR: Required environment variables not set."
    echo "Usage: sbatch --export=ALL,MODEL=<model>,UCB_LAMBDA=<lambda>,SEED=<seed> $0"
    exit 1
fi

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Create logs directory
mkdir -p logs

# Fixed parameters
GENERATIONS=5000
NUM_EMITTERS=64
BATCH_SIZE=8
PARCEL_SIZE=60

echo "Configuration:"
echo "  Model: $MODEL"
echo "  UCB lambda: $UCB_LAMBDA"
echo "  Parcel size: $PARCEL_SIZE"
echo "  Generations: $GENERATIONS"
echo "  Emitters: $NUM_EMITTERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"

# Model paths
UNET_MODEL="results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth"
SVGP_MODEL="results/exp3_hpo/hyperparameterization/model_combined_ind5000_kmeans_rep2.pth"

# Output directory
OUTPUT_DIR="results/exp6_qd_comparison"

# Run optimization
echo "Starting MAP-Elites optimization..."

python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model $MODEL \
    --ucb-lambda $UCB_LAMBDA \
    --unet-model $UNET_MODEL \
    --svgp-model $SVGP_MODEL \
    --parcel-size $PARCEL_SIZE \
    --generations $GENERATIONS \
    --num-emitters $NUM_EMITTERS \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --output-dir $OUTPUT_DIR \
    --no-compile

echo "Job completed at $(date)"
