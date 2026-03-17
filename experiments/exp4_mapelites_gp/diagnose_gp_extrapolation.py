#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Diagnostics: GP Extrapolation Analysis for MAP-Elites Archives

This script diagnoses why MAP-Elites archives validate poorly by checking:
1. Genome distribution shift: How far have archive genomes drifted from training data?
2. GP uncertainty: Does the GP show high uncertainty on archive solutions?

Usage:
    python experiments/exp4_mapelites_gp/diagnose_gp_extrapolation.py \\
        --archive results/exp4_mapelites_gp/mapelites_gp/emit64_batch16_rep1/archive_final.pkl \\
        --gp-model results/exp3_hpo/hyperparameterization/model_combined_ind1000_random_rep1.pth \\
        --training-data results/exp1_gp_training_data/training_datasets/dataset_combined.npz \\
        --output-dir results/exp4_mapelites_gp/diagnostics/emit64_batch16_rep1

For batch processing of all archives:
    sbatch hpc/exp4_mapelites_gp/submit_diagnose_all_archives.sh
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
import yaml
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp3_hpo.train_gp_hpo import SVGPModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Diagnostic 1: Genome Distribution Shift
# ============================================================================

def analyze_genome_distribution_shift(
    archive_genomes: np.ndarray,
    training_genomes: np.ndarray,
    output_dir: Path
) -> Dict:
    """
    Compare genome distributions between archive and training data.
    
    Returns metrics quantifying distribution shift.
    """
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC 1: Genome Distribution Shift")
    logger.info("="*70)
    
    n_dims = archive_genomes.shape[1]
    
    # Compute per-dimension statistics
    archive_mean = archive_genomes.mean(axis=0)
    archive_std = archive_genomes.std(axis=0)
    archive_min = archive_genomes.min(axis=0)
    archive_max = archive_genomes.max(axis=0)
    
    training_mean = training_genomes.mean(axis=0)
    training_std = training_genomes.std(axis=0)
    training_min = training_genomes.min(axis=0)
    training_max = training_genomes.max(axis=0)
    
    # Compute shift metrics
    mean_shift = np.abs(archive_mean - training_mean)
    std_ratio = archive_std / (training_std + 1e-8)
    
    # Count dimensions with significant shift (>2 std devs)
    z_score_shift = mean_shift / (training_std + 1e-8)
    n_shifted_dims = (z_score_shift > 2.0).sum()
    
    # KS test for distribution similarity (sample 10 dimensions)
    sample_dims = np.linspace(0, n_dims-1, min(10, n_dims), dtype=int)
    ks_stats = []
    ks_pvals = []
    for dim in sample_dims:
        ks_stat, ks_pval = stats.ks_2samp(
            training_genomes[:, dim], 
            archive_genomes[:, dim]
        )
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_pval)
    
    mean_ks_stat = np.mean(ks_stats)
    mean_ks_pval = np.mean(ks_pvals)
    
    # Check for out-of-range values
    n_below_min = (archive_genomes < training_min).sum(axis=0)
    n_above_max = (archive_genomes > training_max).sum(axis=0)
    total_oor = n_below_min.sum() + n_above_max.sum()
    total_values = archive_genomes.size
    oor_fraction = total_oor / total_values
    
    metrics = {
        'n_archive_samples': len(archive_genomes),
        'n_training_samples': len(training_genomes),
        'n_dimensions': n_dims,
        'mean_abs_shift': float(mean_shift.mean()),
        'max_abs_shift': float(mean_shift.max()),
        'mean_std_ratio': float(std_ratio.mean()),
        'n_shifted_dims_2sigma': int(n_shifted_dims),
        'fraction_shifted_dims': float(n_shifted_dims / n_dims),
        'mean_ks_statistic': float(mean_ks_stat),
        'mean_ks_pvalue': float(mean_ks_pval),
        'out_of_range_fraction': float(oor_fraction),
        'total_out_of_range_values': int(total_oor),
    }
    
    logger.info(f"\nDistribution Shift Metrics:")
    logger.info(f"  Archive samples:        {metrics['n_archive_samples']:,}")
    logger.info(f"  Training samples:       {metrics['n_training_samples']:,}")
    logger.info(f"  Mean absolute shift:    {metrics['mean_abs_shift']:.4f}")
    logger.info(f"  Max absolute shift:     {metrics['max_abs_shift']:.4f}")
    logger.info(f"  Mean std ratio:         {metrics['mean_std_ratio']:.4f}")
    logger.info(f"  Dims shifted >2σ:       {metrics['n_shifted_dims_2sigma']}/{n_dims} ({100*metrics['fraction_shifted_dims']:.1f}%)")
    logger.info(f"  Mean KS statistic:      {metrics['mean_ks_statistic']:.4f}")
    logger.info(f"  Mean KS p-value:        {metrics['mean_ks_pvalue']:.2e}")
    logger.info(f"  Out-of-range fraction:  {100*metrics['out_of_range_fraction']:.2f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Mean shift per dimension
    ax = axes[0, 0]
    ax.bar(range(n_dims), mean_shift, alpha=0.7)
    ax.axhline(2 * training_std.mean(), color='r', linestyle='--', label='2σ threshold')
    ax.set_xlabel('Genome Dimension')
    ax.set_ylabel('|Mean Shift|')
    ax.set_title(f'Mean Shift per Dimension\n({n_shifted_dims}/{n_dims} exceed 2σ)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Std ratio per dimension
    ax = axes[0, 1]
    ax.bar(range(n_dims), std_ratio, alpha=0.7)
    ax.axhline(1.0, color='r', linestyle='--', label='Equal variance')
    ax.set_xlabel('Genome Dimension')
    ax.set_ylabel('Std Ratio (Archive / Training)')
    ax.set_title('Variance Change per Dimension')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Range comparison (sample 10 dims)
    ax = axes[0, 2]
    sample_dims_plot = np.linspace(0, n_dims-1, 10, dtype=int)
    x = np.arange(len(sample_dims_plot))
    width = 0.35
    ax.bar(x - width/2, training_max[sample_dims_plot] - training_min[sample_dims_plot], 
           width, label='Training range', alpha=0.7)
    ax.bar(x + width/2, archive_max[sample_dims_plot] - archive_min[sample_dims_plot], 
           width, label='Archive range', alpha=0.7)
    ax.set_xlabel('Dimension (sampled)')
    ax.set_ylabel('Range')
    ax.set_title('Range Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_dims_plot)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4-6. Distribution overlays for 3 representative dimensions
    dims_to_plot = [0, n_dims//2, n_dims-1]
    for i, dim in enumerate(dims_to_plot):
        ax = axes[1, i]
        ax.hist(training_genomes[:, dim], bins=50, alpha=0.5, label='Training', density=True)
        ax.hist(archive_genomes[:, dim], bins=50, alpha=0.5, label='Archive', density=True)
        ax.set_xlabel(f'Genome value (dim {dim})')
        ax.set_ylabel('Density')
        ax.set_title(f'Dimension {dim}\nKS={ks_stats[i]:.3f}, shift={mean_shift[dim]:.2f}' 
                     if i < len(ks_stats) else f'Dimension {dim}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic1_genome_distribution_shift.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'diagnostic1_genome_distribution_shift.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved plots to {output_dir}")
    
    return metrics


# ============================================================================
# Diagnostic 2: GP Uncertainty Analysis
# ============================================================================

def analyze_gp_uncertainty(
    archive_solutions: np.ndarray,
    test_solutions: np.ndarray,
    gp_model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    output_dir: Path
) -> Dict:
    """
    Compare GP predictive uncertainty on archive vs validation data.
    
    Returns metrics quantifying extrapolation risk.
    """
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC 2: GP Uncertainty Analysis")
    logger.info("="*70)
    
    device = next(gp_model.parameters()).device
    
    # Predict on archive solutions
    logger.info("Computing GP predictions on archive solutions...")
    archive_x = torch.tensor(archive_solutions, dtype=torch.float32).to(device)
    archive_x_norm = (archive_x - gp_model.train_x_mean) / (gp_model.train_x_std + 1e-6)
    
    gp_model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        archive_pred = likelihood(gp_model(archive_x_norm))
        archive_mean = (archive_pred.mean * gp_model.train_y_std + gp_model.train_y_mean).cpu().numpy()
        archive_std = (archive_pred.stddev * gp_model.train_y_std).cpu().numpy()
    
    # Predict on validation solutions
    logger.info("Computing GP predictions on validation solutions...")
    test_x = torch.tensor(test_solutions, dtype=torch.float32).to(device)
    test_x_norm = (test_x - gp_model.train_x_mean) / (gp_model.train_x_std + 1e-6)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_pred = likelihood(gp_model(test_x_norm))
        test_mean = (test_pred.mean * gp_model.train_y_std + gp_model.train_y_mean).cpu().numpy()
        test_std = (test_pred.stddev * gp_model.train_y_std).cpu().numpy()
    
    # Compute coefficient of variation (CV = std/mean)
    archive_cv = archive_std / (np.abs(archive_mean) + 1e-6)
    test_cv = test_std / (np.abs(test_mean) + 1e-6)
    
    # Compute metrics
    metrics = {
        'archive_mean_uncertainty': float(archive_std.mean()),
        'archive_median_uncertainty': float(np.median(archive_std)),
        'archive_max_uncertainty': float(archive_std.max()),
        'archive_mean_cv': float(archive_cv.mean()),
        'val_mean_uncertainty': float(test_std.mean()),
        'val_median_uncertainty': float(np.median(test_std)),
        'val_max_uncertainty': float(test_std.max()),
        'val_mean_cv': float(test_cv.mean()),
        'uncertainty_ratio': float(archive_std.mean() / test_std.mean()),
        'cv_ratio': float(archive_cv.mean() / test_cv.mean()),
        'fraction_archive_higher_unc': float((archive_std > test_std.mean()).mean()),
        'archive_mean_prediction': float(archive_mean.mean()),
        'val_mean_prediction': float(test_mean.mean()),
    }
    
    logger.info(f"\nUncertainty Metrics:")
    logger.info(f"  Archive:")
    logger.info(f"    Mean uncertainty:       {metrics['archive_mean_uncertainty']:.4f}")
    logger.info(f"    Median uncertainty:     {metrics['archive_median_uncertainty']:.4f}")
    logger.info(f"    Max uncertainty:        {metrics['archive_max_uncertainty']:.4f}")
    logger.info(f"    Mean CV:                {metrics['archive_mean_cv']:.4f}")
    logger.info(f"    Mean prediction:        {metrics['archive_mean_prediction']:.4f}")
    logger.info(f"  Validation data:")
    logger.info(f"    Mean uncertainty:       {metrics['val_mean_uncertainty']:.4f}")
    logger.info(f"    Median uncertainty:     {metrics['val_median_uncertainty']:.4f}")
    logger.info(f"    Max uncertainty:        {metrics['val_max_uncertainty']:.4f}")
    logger.info(f"    Mean CV:                {metrics['val_mean_cv']:.4f}")
    logger.info(f"    Mean prediction:        {metrics['val_mean_prediction']:.4f}")
    logger.info(f"  Comparison:")
    logger.info(f"    Uncertainty ratio:      {metrics['uncertainty_ratio']:.4f}x")
    logger.info(f"    CV ratio:               {metrics['cv_ratio']:.4f}x")
    logger.info(f"    Archive > test mean:    {100*metrics['fraction_archive_higher_unc']:.1f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Uncertainty distributions
    ax = axes[0, 0]
    ax.hist(test_std, bins=50, alpha=0.5, label=f'Validation (mean={test_std.mean():.2f})', density=True)
    ax.hist(archive_std, bins=50, alpha=0.5, label=f'Archive (mean={archive_std.mean():.2f})', density=True)
    ax.axvline(test_std.mean(), color='C0', linestyle='--', alpha=0.7)
    ax.axvline(archive_std.mean(), color='C1', linestyle='--', alpha=0.7)
    ax.set_xlabel('Predictive Std Dev')
    ax.set_ylabel('Density')
    ax.set_title(f'Uncertainty Distribution\nRatio={metrics["uncertainty_ratio"]:.2f}x')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. CV distributions
    ax = axes[0, 1]
    ax.hist(test_cv, bins=50, alpha=0.5, label=f'Validation (mean={test_cv.mean():.2f})', density=True, range=(0, 1))
    ax.hist(archive_cv, bins=50, alpha=0.5, label=f'Archive (mean={archive_cv.mean():.2f})', density=True, range=(0, 1))
    ax.set_xlabel('Coefficient of Variation (std/mean)')
    ax.set_ylabel('Density')
    ax.set_title(f'Relative Uncertainty\nRatio={metrics["cv_ratio"]:.2f}x')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Uncertainty vs prediction (archive)
    ax = axes[1, 0]
    scatter = ax.scatter(archive_mean, archive_std, alpha=0.5, s=20, c=np.arange(len(archive_mean)), cmap='viridis')
    ax.set_xlabel('GP Mean Prediction')
    ax.set_ylabel('GP Std Dev')
    ax.set_title('Archive: Uncertainty vs Prediction')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sample index')
    
    # 4. Uncertainty vs prediction (validation)
    ax = axes[1, 1]
    scatter = ax.scatter(test_mean, test_std, alpha=0.5, s=20, c=np.arange(len(test_mean)), cmap='viridis')
    ax.set_xlabel('GP Mean Prediction')
    ax.set_ylabel('GP Std Dev')
    ax.set_title('Validation: Uncertainty vs Prediction')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sample index')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic2_gp_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'diagnostic2_gp_uncertainty.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved plots to {output_dir}")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose GP extrapolation in MAP-Elites archives"
    )
    parser.add_argument(
        "--archive",
        type=str,
        required=True,
        help="Path to archive pickle file"
    )
    parser.add_argument(
        "--gp-model",
        type=str,
        required=True,
        help="Path to trained GP model (.pth file)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to training data NPZ file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: archive_dir/diagnostics)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU"
    )
    args = parser.parse_args()
    
    # Setup paths
    archive_path = Path(args.archive)
    gp_model_path = Path(args.gp_model)
    training_data_path = Path(args.training_data)
    
    if args.output_dir is None:
        output_dir = archive_path.parent / "diagnostics"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("GP Extrapolation Diagnostics")
    logger.info("="*70)
    logger.info(f"Archive:       {archive_path}")
    logger.info(f"GP model:      {gp_model_path}")
    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Output:        {output_dir}")
    logger.info("")
    
    # Load archive
    logger.info("Loading archive...")
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    
    archive_data = archive.data()
    archive_solutions = np.array(archive_data['solution'])  # (N, 62)
    archive_genomes = archive_solutions[:, :60]  # Extract genomes only
    
    logger.info(f"  Archive elites: {len(archive_solutions):,}")
    
    # Load training data
    logger.info("Loading training data...")
    data = np.load(training_data_path)
    
    # Use train data for distribution comparison
    training_genomes = data['train_genomes']
    
    # Use val data as "test" set for GP uncertainty comparison
    # (the prepared datasets only have train/val splits, no separate test)
    test_genomes = data['val_genomes']
    test_widths = data['val_widths']
    test_heights = data['val_heights']
    
    logger.info(f"  Training samples: {len(training_genomes):,}")
    logger.info(f"  Validation samples (for GP uncertainty): {len(test_genomes):,}")
    
    # Prepare full solutions for GP (with parcel size)
    archive_parcel_size = archive_solutions[0, 60]  # Should be constant
    test_solutions = np.column_stack([
        test_genomes,
        test_widths.reshape(-1, 1),
        test_heights.reshape(-1, 1)
    ]).astype(np.float32)
    
    logger.info(f"  Archive parcel size: {archive_parcel_size}m")
    
    # Diagnostic 1: Genome distribution shift
    shift_metrics = analyze_genome_distribution_shift(
        archive_genomes,
        training_genomes,
        output_dir
    )
    
    # Load GP model
    logger.info("\nLoading GP model...")
    device = torch.device('cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"  Device: {device}")
    
    checkpoint = torch.load(gp_model_path, map_location=device)
    
    inducing_key = 'variational_strategy.inducing_points'
    inducing_points = checkpoint['model_state_dict'][inducing_key]
    num_inducing = inducing_points.size(0)
    input_dim = inducing_points.size(1)
    
    model = SVGPModel(inducing_points.to(device), input_dim=input_dim).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    model.train_x_mean = checkpoint['train_x_mean'].to(device)
    model.train_x_std = checkpoint['train_x_std'].to(device)
    model.train_y_mean = checkpoint['train_y_mean'].to(device)
    model.train_y_std = checkpoint['train_y_std'].to(device)
    
    model.eval()
    likelihood.eval()
    
    logger.info(f"  Model loaded: {num_inducing} inducing points, {input_dim}D input")
    
    # Diagnostic 2: GP uncertainty
    uncertainty_metrics = analyze_gp_uncertainty(
        archive_solutions,
        test_solutions,
        model,
        likelihood,
        output_dir
    )
    
    # Combine metrics and save
    all_metrics = {
        'archive_path': str(archive_path),
        'gp_model_path': str(gp_model_path),
        'training_data_path': str(training_data_path),
        'distribution_shift': shift_metrics,
        'uncertainty_analysis': uncertainty_metrics,
    }
    
    metrics_path = output_dir / 'diagnostic_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Diagnostics complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - diagnostic1_genome_distribution_shift.png/pdf")
    logger.info(f"  - diagnostic2_gp_uncertainty.png/pdf")
    logger.info(f"  - diagnostic_metrics.yaml")
    logger.info("")
    
    # Print summary interpretation
    logger.info("INTERPRETATION:")
    logger.info("-" * 70)
    
    if shift_metrics['fraction_shifted_dims'] > 0.5:
        logger.info("⚠ SIGNIFICANT DISTRIBUTION SHIFT DETECTED")
        logger.info(f"  {100*shift_metrics['fraction_shifted_dims']:.0f}% of dimensions shifted >2σ from training")
    else:
        logger.info("✓ Genome distribution relatively stable")
    
    if shift_metrics['out_of_range_fraction'] > 0.1:
        logger.info(f"⚠ {100*shift_metrics['out_of_range_fraction']:.1f}% of archive values outside training range")
    
    if uncertainty_metrics['uncertainty_ratio'] > 1.5:
        logger.info(f"⚠ HIGH EXTRAPOLATION RISK: Archive uncertainty {uncertainty_metrics['uncertainty_ratio']:.1f}× higher than validation")
    elif uncertainty_metrics['uncertainty_ratio'] > 1.1:
        logger.info(f"⚠ Moderate extrapolation risk: Archive uncertainty {uncertainty_metrics['uncertainty_ratio']:.1f}× higher")
    else:
        logger.info(f"✓ Archive uncertainty similar to validation ({uncertainty_metrics['uncertainty_ratio']:.2f}×)")
    
    if uncertainty_metrics['archive_mean_prediction'] > 45 and uncertainty_metrics['val_mean_prediction'] < 35:
        logger.info(f"⚠ SATURATION DETECTED: Archive predictions ({uncertainty_metrics['archive_mean_prediction']:.1f}) much higher than validation ({uncertainty_metrics['val_mean_prediction']:.1f})")
        logger.info("  → GP likely saturated in false optimum region")
    
    logger.info("")


if __name__ == "__main__":
    main()
