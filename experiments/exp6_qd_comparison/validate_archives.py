#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Validate Archive Solutions with Real KLAM_21

After MAP-Elites optimization with offline surrogates, this script validates
all archive solutions by re-evaluating them with the real KLAM_21 physics simulation.

This provides ground-truth objectives for computing:
- True QD scores
- Prediction accuracy metrics (R², RMSE, Spearman ρ)
- Calibration analysis

Usage:
    python experiments/exp6_qd_comparison/validate_archives.py \
        --archive archive_unet_size51_seed42.pkl \
        --output-dir results/exp6_qd_comparison/validation
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from domain_description import evaluation_klam

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_archive(archive_path: Path, output_dir: Path, parcel_size: int, max_solutions: int = None):
    """
    Validate archive solutions with real KLAM_21.
    
    Args:
        archive_path: Path to archive .pkl file
        output_dir: Output directory for validation results
        parcel_size: Parcel size in meters (e.g., 60)
        max_solutions: Maximum number of solutions to validate (for testing)
    """
    logger.info("="*80)
    logger.info("ARCHIVE VALIDATION WITH KLAM_21")
    logger.info("="*80)
    logger.info(f"Archive: {archive_path}")
    logger.info(f"Parcel size: {parcel_size}m")
    
    # Load archive
    logger.info("Loading archive...")
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    
    # Extract solutions
    data = archive.data(fields=['solution', 'measures', 'objective', 'index'])
    solutions = data['solution']
    measures = data['measures']
    objectives_predicted = data['objective']
    indices_archive = data['index']
    
    n_solutions = len(solutions)
    logger.info(f"  Archive size: {n_solutions} solutions")
    logger.info(f"  Archive coverage: {archive.stats.coverage:.2%}")
    logger.info(f"  QD score (predicted): {archive.stats.qd_score:.2f}")
    
    # Stratified sampling for diversity (one solution per cell, up to max_solutions)
    if max_solutions is not None and n_solutions > max_solutions:
        logger.info(f"  Sampling {max_solutions} diverse solutions (one per cell)...")
        
        # Get unique cell indices
        unique_cells, cell_first_occurrence = np.unique(indices_archive, return_index=True)
        
        if len(unique_cells) <= max_solutions:
            # If fewer cells than max_solutions, sample all cells
            logger.info(f"    Archive has {len(unique_cells)} cells, sampling all")
            selected_indices = cell_first_occurrence
        else:
            # If more cells than max_solutions, randomly sample cells
            logger.info(f"    Archive has {len(unique_cells)} cells, sampling {max_solutions}")
            sampled_cells = np.random.choice(len(unique_cells), max_solutions, replace=False)
            selected_indices = cell_first_occurrence[sampled_cells]
        
        solutions = solutions[selected_indices]
        measures = measures[selected_indices]
        objectives_predicted = objectives_predicted[selected_indices]
        n_solutions = len(selected_indices)
        
        logger.info(f"  Selected {n_solutions} solutions from {len(selected_indices)} cells")
    
    # Initialize KLAM environment
    logger.info("Initializing KLAM_21 environment...")
    with open(project_root / 'domain_description' / 'cfg.yml') as f:
        config_environment = yaml.safe_load(f)
    
    config_environment = evaluation_klam.init_environment(config_environment)
    
    # Validate each solution
    logger.info("Validating solutions with KLAM_21...")
    objectives_klam = []
    features_klam = []
    
    start_time = time.time()
    
    for i, solution in enumerate(solutions):
        # Solutions are 60D genomes only (parcel size is fixed per experiment)
        genome = solution[:60] if len(solution) >= 60 else solution
        
        # Update environment config for this parcel size
        config_environment['parcel_size_m'] = int(parcel_size)
        
        # Create encoding for this parcel size
        with open(project_root / 'encodings' / 'parametric' / 'cfg.yml') as f:
            config_encoding = yaml.safe_load(f)
        
        config_encoding['length_design'] = int(parcel_size / 3)
        config_encoding['parcel_width_m'] = float(parcel_size)
        config_encoding['parcel_height_m'] = float(parcel_size)
        
        solution_template = ParametricEncoding(config=config_encoding)
        
        # Evaluate with KLAM_21
        try:
            result_array, debug_data, _ = evaluation_klam.eval(
                genome, 
                config_environment, 
                config_encoding,
                solution_template,
                use_surrogate=False,
                debug=False,
                collect_spatial_data=False
            )
            # Extract fitness (first value) and features (next 8 values)
            obj = result_array[0]
            feat = result_array[1:9]
            objectives_klam.append(obj)
            features_klam.append(feat)
        except Exception as e:
            logger.error(f"  Error evaluating solution {i}: {e}")
            objectives_klam.append(np.nan)
            features_klam.append(np.full(8, np.nan))
        
        # Progress logging
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (n_solutions - i - 1)
            logger.info(
                f"  Progress: {i+1}/{n_solutions} "
                f"({(i+1)/n_solutions*100:.1f}%) | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {remaining/60:.1f}min"
            )
    
    objectives_klam = np.array(objectives_klam)
    features_klam = np.array(features_klam)
    
    elapsed = time.time() - start_time
    logger.info(f"Validation complete! Time: {elapsed/60:.1f} min")
    
    # Compute validation metrics
    logger.info("="*80)
    logger.info("VALIDATION METRICS")
    logger.info("="*80)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(objectives_klam)
    n_valid = np.sum(valid_mask)
    n_failed = n_solutions - n_valid
    
    logger.info(f"Valid evaluations: {n_valid}/{n_solutions} ({n_valid/n_solutions*100:.1f}%)")
    if n_failed > 0:
        logger.info(f"Failed evaluations: {n_failed}")
    
    # Use only valid solutions for metrics
    obj_pred = objectives_predicted[valid_mask]
    obj_klam = objectives_klam[valid_mask]
    
    # R² score
    ss_res = np.sum((obj_klam - obj_pred) ** 2)
    ss_tot = np.sum((obj_klam - np.mean(obj_klam)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean((obj_klam - obj_pred) ** 2))
    mae = np.mean(np.abs(obj_klam - obj_pred))
    
    # Spearman correlation
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(obj_pred, obj_klam)
    
    # QD scores — matched comparison (same N solutions)
    qd_score_validated = np.sum(obj_klam)        # sum of KLAM objectives for validated subset
    qd_score_predicted_subset = np.sum(obj_pred)  # sum of surrogate objectives for SAME subset
    qd_score_predicted_full = archive.stats.qd_score  # full archive (all elites, for reference)
    n_archive_total = len(archive)                 # total elites in archive
    
    logger.info(f"R² score: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Spearman ρ: {rho:.4f} (p={p_value:.2e})")
    logger.info(f"QD score (predicted subset, N={n_valid}): {qd_score_predicted_subset:.2f}")
    logger.info(f"QD score (validated subset,  N={n_valid}): {qd_score_validated:.2f}")
    logger.info(f"QD score ratio (matched): {qd_score_validated/qd_score_predicted_subset:.3f}")
    logger.info(f"QD score (full archive, N={n_archive_total}): {qd_score_predicted_full:.2f}")
    
    # Save validation results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as NPZ
    output_name = archive_path.stem + "_validated"
    npz_path = output_dir / f"{output_name}.npz"
    
    logger.info(f"Saving validation results to {npz_path}")
    np.savez(
        npz_path,
        solutions=solutions,
        measures=measures,
        objectives_predicted=objectives_predicted,
        objectives_klam=objectives_klam,
        features_klam=features_klam,
        valid_mask=valid_mask,
    )
    
    # Save metrics as JSON
    metrics = {
        'archive_path': str(archive_path),
        'n_solutions': int(n_solutions),
        'n_valid': int(n_valid),
        'n_failed': int(n_failed),
        'validation_time_minutes': float(elapsed / 60),
        'metrics': {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'spearman_rho': float(rho),
            'spearman_p': float(p_value),
        },
        'qd_scores': {
            'predicted_subset': float(qd_score_predicted_subset),
            'validated_subset': float(qd_score_validated),
            'ratio': float(qd_score_validated / qd_score_predicted_subset),
            'predicted_full_archive': float(qd_score_predicted_full),
        },
        'n_archive_total': int(n_archive_total),
        'n_compared': int(n_valid),
        'archive_stats': {
            'coverage': float(archive.stats.coverage),
        }
    }
    
    json_path = output_dir / f"{output_name}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {json_path}")
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Validate archive with KLAM_21")
    
    parser.add_argument('--archive', type=str, required=True,
                       help='Path to archive .pkl file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--parcel-size', type=int, default=None,
                       help='Parcel size in meters (default: auto-detect from filename)')
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum number of solutions to validate (for testing)')
    
    args = parser.parse_args()
    
    # Extract parcel size from filename if not provided
    parcel_size = args.parcel_size
    if parcel_size is None:
        # Try to extract from filename pattern: archive_MODEL_size27_seedXX.pkl
        import re
        match = re.search(r'size(\d+)', Path(args.archive).name)
        if match:
            parcel_size = int(match.group(1))
            logger.info(f"Auto-detected parcel size: {parcel_size}m from filename")
        else:
            raise ValueError(
                "Could not auto-detect parcel size from filename. "
                "Please provide --parcel-size argument."
            )
    
    validate_archive(
        Path(args.archive),
        Path(args.output_dir),
        parcel_size,
        args.max_solutions
    )


if __name__ == '__main__':
    main()
