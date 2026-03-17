#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Validate MAP-Elites Archive with Real KLAM_21

This script loads a MAP-Elites archive and re-evaluates the best solutions
with the real KLAM_21 physics simulation to validate GP predictions.

Usage:
    python experiments/validate_mapelites_archive.py --archive results/mapelites_gp/emit64_batch16_rep1/archive_final.pkl
    python experiments/validate_mapelites_archive.py --archive results/mapelites_gp/emit64_batch16_rep1/archive_final.pkl --top-n 100
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from domain_description import evaluation_klam
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_archive(archive_path: Path):
    """Load pickled archive."""
    logger.info(f"Loading archive from {archive_path}")
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    
    stats = archive.stats
    data = archive.data()
    logger.info(f"  Archive stats:")
    logger.info(f"    Coverage: {stats.coverage:.2%}")
    logger.info(f"    QD Score: {stats.qd_score:.1f}")
    logger.info(f"    Max Objective: {stats.obj_max:.2f}")
    logger.info(f"    Num Elites: {len(data['solution'])}")
    
    return archive


def select_solutions_for_validation(
    archive,
    n_samples: int = 100,
    strategy: str = 'top'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select solutions from archive for validation.
    
    Args:
        archive: PyRibs GridArchive
        n_samples: Number of solutions to validate
        strategy: 'top' (best objectives) or 'random' or 'diverse'
    
    Returns:
        solutions: (N, 62) array
        gp_objectives: (N,) array of GP predictions
        indices: (N,) array of selected indices
    """
    data = archive.data()
    n_elites = len(data['solution'])
    
    solutions = np.array(data['solution'])
    gp_objectives = np.array(data['objective'])
    indices = np.array(data['index'])
    
    if strategy == 'top':
        # Select top N by objective
        top_indices = np.argsort(gp_objectives)[-min(n_samples, n_elites):][::-1]
        solutions = solutions[top_indices]
        gp_objectives = gp_objectives[top_indices]
        indices = indices[top_indices]
    elif strategy == 'random':
        # Random sample
        random_indices = np.random.choice(n_elites, min(n_samples, n_elites), replace=False)
        solutions = solutions[random_indices]
        gp_objectives = gp_objectives[random_indices]
        indices = indices[random_indices]
    elif strategy == 'diverse':
        # Stratified sampling across feature space
        # Sample first elite from each unique bin, up to n_samples
        unique_indices, first_occurrences = np.unique(indices, return_index=True)
        sample_indices = first_occurrences[:n_samples]
        solutions = solutions[sample_indices]
        gp_objectives = gp_objectives[sample_indices]
        indices = indices[sample_indices]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.info(f"Selected {len(solutions)} solutions using '{strategy}' strategy")
    
    return solutions, gp_objectives, indices


def evaluate_with_klam(solutions: np.ndarray, config_environment: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate solutions with real KLAM_21.
    
    Args:
        solutions: (N, 62) array of [genome(60), parcel_width, parcel_height]
        config_environment: KLAM configuration
    
    Returns:
        objectives: (N,) array of KLAM objectives
        features: (N, 8) array of planning features
    """
    objectives = []
    features_list = []
    
    # Get number of features from config
    n_features = len(config_environment.get('features', [0,1,2,3,4,5,6,7]))
    
    for i, sol in enumerate(solutions):
        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluating solution {i+1}/{len(solutions)}...")
        
        # Extract genome and parcel size
        genome = sol[:60]
        parcel_size = int(sol[60])  # Width and height are same
        length_design = parcel_size // 3
        
        # Create encoding config and template
        config_encoding = {
            'length_design': length_design,
            'max_num_buildings': 10,
            'max_num_floors': 10,
            'xy_scale': 3.0,
            'z_scale': 3.0
        }
        solution_template = ParametricEncoding(config=config_encoding)
        
        # Evaluate with KLAM using new API
        # eval() returns (result_array, debug_data, spatial_data)
        # result_array = [fitness, features..., phenotype_floors...]
        result_array, _, _ = evaluation_klam.eval(
            genome, config_environment, config_encoding, solution_template
        )
        
        # Extract fitness (first element) and features (next n_features elements)
        obj = result_array[0]
        feats = result_array[1:1+n_features]
        
        objectives.append(obj)
        features_list.append(feats)
    
    return np.array(objectives), np.array(features_list)


def compute_validation_metrics(gp_pred: np.ndarray, klam_true: np.ndarray) -> Dict:
    """Compute validation metrics."""
    # Basic metrics
    mse = np.mean((gp_pred - klam_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(gp_pred - klam_true))
    
    # R²
    ss_res = np.sum((klam_true - gp_pred)**2)
    ss_tot = np.sum((klam_true - np.mean(klam_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    
    # Correlation
    pearson_r, pearson_p = pearsonr(gp_pred, klam_true)
    spearman_rho, spearman_p = spearmanr(gp_pred, klam_true)
    
    # Relative error
    rel_error = np.abs(gp_pred - klam_true) / (np.abs(klam_true) + 1e-6)
    mean_rel_error = np.mean(rel_error)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'mean_relative_error': float(mean_rel_error),
    }


def plot_validation_results(
    gp_pred: np.ndarray,
    klam_true: np.ndarray,
    metrics: Dict,
    output_dir: Path
):
    """Create validation plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(klam_true, gp_pred, alpha=0.6, s=50)
    
    # Identity line
    min_val = min(klam_true.min(), gp_pred.min())
    max_val = max(klam_true.max(), gp_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('KLAM_21 True Objective')
    ax.set_ylabel('GP Predicted Objective')
    ax.set_title(f"GP vs KLAM_21\nR²={metrics['r2']:.3f}, ρ={metrics['spearman_rho']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Residual plot
    ax = axes[1]
    residuals = gp_pred - klam_true
    ax.scatter(klam_true, residuals, alpha=0.6, s=50)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('KLAM_21 True Objective')
    ax.set_ylabel('Residual (GP - KLAM)')
    ax.set_title(f"Residuals\nRMSE={metrics['rmse']:.2f}")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_scatter.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'validation_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved validation plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MAP-Elites archive with real KLAM_21"
    )
    parser.add_argument(
        "--archive",
        type=str,
        required=True,
        help="Path to archive pickle file"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top solutions to validate"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="top",
        choices=['top', 'random', 'diverse'],
        help="Solution selection strategy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as archive)"
    )
    args = parser.parse_args()
    
    # Setup paths
    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    if args.output_dir is None:
        output_dir = archive_path.parent / "validation"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MAP-Elites Archive Validation")
    logger.info("=" * 70)
    logger.info(f"Archive: {archive_path}")
    logger.info(f"Validating top {args.top_n} solutions")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # Load archive
    archive = load_archive(archive_path)
    
    # Select solutions
    logger.info("\nSelecting solutions for validation...")
    solutions, gp_objectives, indices = select_solutions_for_validation(
        archive, n_samples=args.top_n, strategy=args.strategy
    )
    
    # Load KLAM config
    config_path = project_root / "domain_description" / "cfg.yml"
    with open(config_path) as f:
        config_environment = yaml.safe_load(f)
    
    config_environment = evaluation_klam.init_environment(config_environment)
    
    # Evaluate with KLAM
    logger.info("\nEvaluating solutions with KLAM_21...")
    logger.info("(This may take a while...)")
    klam_objectives, klam_features = evaluate_with_klam(solutions, config_environment)
    
    # Compute metrics
    logger.info("\nComputing validation metrics...")
    metrics = compute_validation_metrics(gp_objectives, klam_objectives)
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"R² Score:         {metrics['r2']:.4f}")
    logger.info(f"RMSE:             {metrics['rmse']:.4f}")
    logger.info(f"MAE:              {metrics['mae']:.4f}")
    logger.info(f"Pearson r:        {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    logger.info(f"Spearman ρ:       {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.2e})")
    logger.info(f"Mean Rel. Error:  {metrics['mean_relative_error']:.4f}")
    logger.info("")
    
    # Save results
    results_df = pd.DataFrame({
        'index': indices,
        'gp_objective': gp_objectives,
        'klam_objective': klam_objectives,
        'residual': gp_objectives - klam_objectives,
        'relative_error': np.abs(gp_objectives - klam_objectives) / (np.abs(klam_objectives) + 1e-6)
    })
    
    results_path = output_dir / 'validation_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Save metrics
    metrics_path = output_dir / 'validation_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_validation_results(gp_objectives, klam_objectives, metrics, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Validation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
