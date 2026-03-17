#!/bin/bash
#SBATCH --job-name=diagnose_coverage
#SBATCH --output=logs/diagnose_coverage_%j.out
#SBATCH --error=logs/diagnose_coverage_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=hpc

# Diagnose archive coverage discrepancy between Exp 1 and Exp 6

set -e

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate environment
source ~/.bashrc
conda activate openskizze_klam_qd

# Run diagnostic
python tests/diagnose_archive_coverage.py

echo "Job completed at $(date)"
