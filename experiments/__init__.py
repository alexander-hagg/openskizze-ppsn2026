# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
GP Training Data Comparison Experiment

This package contains scripts for running the experiment comparing
GP surrogates trained on different data sources:

1. Optimized (SAIL): Elite solutions from SAIL optimization
2. Random (Sobol): Random samples from scrambled Sobol sequences
3. Combined: 50% Optimized + 50% Random

Experiment Phases:
------------------
- Phase 0: test_klam_evaluation.py - Validate KLAM_21 returns valid values
- Phase 1A: generate_sail_data.py - Generate SAIL training data
- Phase 1B: generate_random_data.py - Generate random training data  
- Phase 2: prepare_training_datasets.py - Combine and balance datasets
- Phase 3: train_gp_experiment.py - Train SVGP models
- Phase 4: evaluate_gp_experiment.py - Cross-domain evaluation

Usage:
------
See EXPERIMENT_PLAN_GP_TRAINING.md for detailed documentation.

Quick start on HPC:
    bash hpc/run_gp_experiment.sh status   # Check progress
    bash hpc/run_gp_experiment.sh 1        # Submit Phase 1
    bash hpc/run_gp_experiment.sh 2        # Run Phase 2
    bash hpc/run_gp_experiment.sh 3        # Submit Phase 3
    bash hpc/run_gp_experiment.sh 4        # Submit Phase 4

Local testing:
    python experiments/test_klam_evaluation.py --num-samples 2 --parcel-size 51
"""
