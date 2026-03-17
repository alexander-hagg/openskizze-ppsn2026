#!/bin/bash
#SBATCH --job-name=validate_features
#SBATCH --output=logs/validate_features_%j.out
#SBATCH --error=logs/validate_features_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=hpc

# Validate feature calculation consistency between Exp 1 and Exp 6

set -e

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Run validation
python tests/validate_feature_calculation.py

echo "Job completed at $(date)"
