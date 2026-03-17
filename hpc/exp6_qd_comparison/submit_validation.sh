#!/bin/bash
#SBATCH --job-name=exp6_validate
#SBATCH --output=logs/exp6_validate_%j.out
#SBATCH --error=logs/exp6_validate_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpc

# Experiment 6: Validate a single archive with KLAM_21 (env-var based)
#
# Required environment variables (pass via sbatch --export=ALL,...):
#   ARCHIVE_PATH  - path to the archive .pkl file
#
# Example:
#   sbatch --export=ALL,ARCHIVE_PATH=results/exp6_qd_comparison/archive_unet_ucb0.0_seed42.pkl \
#       hpc/exp6_qd_comparison/submit_validation.sh

set -e

# Validate required env vars
if [ -z "$ARCHIVE_PATH" ]; then
    echo "ERROR: ARCHIVE_PATH not set."
    echo "Usage: sbatch --export=ALL,ARCHIVE_PATH=<path> $0"
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

VALIDATION_DIR="results/exp6_qd_comparison/validation"

echo "Validating archive: $ARCHIVE_PATH"

# Run validation (100 diverse solutions per archive for faster validation)
python experiments/exp6_qd_comparison/validate_archives.py \
    --archive $ARCHIVE_PATH \
    --output-dir $VALIDATION_DIR \
    --max-solutions 100

echo "Job completed at $(date)"
