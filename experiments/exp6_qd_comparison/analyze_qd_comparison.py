#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Analyze Experiment 6 QD Comparison Results

This script analyzes the results of MAP-Elites optimization with different
offline surrogates and generates comparison plots and tables.

Usage:
    python experiments/exp6_qd_comparison/analyze_qd_comparison.py \
        --results-dir results/exp6_qd_comparison \
        --output-dir results/exp6_qd_comparison/analysis
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


def load_validation_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all validation results from directory."""
    
    results = {}
    
    # Find all validation .npz files
    for npz_path in results_dir.glob('*_validated.npz'):
        # Load data
        data = np.load(npz_path, allow_pickle=True)
        
        # Load metrics
        json_path = npz_path.parent / (npz_path.stem + '_metrics.json')
        with open(json_path) as f:
            metrics = json.load(f)
        
        # Parse configuration from filename
        # Format: archive_<model>_[ucb<lambda>_]size<size>_seed<seed>_validated
        filename = npz_path.stem.replace('_validated', '').replace('archive_', '')
        
        results[filename] = {
            'data': data,
            'metrics': metrics,
            'filename': filename,
        }
    
    return results


def compute_diversity_metrics(solutions: np.ndarray) -> Dict:
    """
    Compute phenotypic diversity metrics.
    
    Args:
        solutions: (N, 62) array of solutions (60D genome + 2D parcel size)
    
    Returns:
        Dictionary of diversity metrics
    """
    # Use genome space (60D)
    genomes = solutions[:, :60]
    
    # Compute pairwise distances
    distances = pdist(genomes, metric='euclidean')
    
    # Basic statistics
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    min_dist = float(np.min(distances)) if len(distances) > 0 else 0.0
    max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
    
    # Solow-Polasky diversity
    theta = 1.0 / (mean_dist + 1e-6)
    dist_matrix = squareform(distances)
    sp_diversity = float(np.sum(np.exp(-theta * dist_matrix)))
    
    return {
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'solow_polasky_diversity': sp_diversity,
    }


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table across all configurations."""
    
    rows = []
    
    for config_name, result in results.items():
        metrics = result['metrics']
        data = result['data']
        
        # Filter valid solutions
        valid_mask = data['valid_mask']
        solutions = data['solutions'][valid_mask]
        obj_pred = data['objectives_predicted'][valid_mask]
        obj_klam = data['objectives_klam'][valid_mask]
        
        # Compute diversity
        diversity = compute_diversity_metrics(solutions)
        
        # Parse configuration
        parts = config_name.split('_')
        model_type = parts[0]
        ucb_lambda = 0.0
        for part in parts:
            if part.startswith('ucb'):
                ucb_lambda = float(part.replace('ucb', ''))
        
        row = {
            'Configuration': config_name,
            'Model': model_type,
            'UCB λ': ucb_lambda,
            'QD Score (Predicted)': metrics['qd_scores'].get('predicted_subset', metrics['qd_scores'].get('predicted', 0)),
            'QD Score (Validated)': metrics['qd_scores'].get('validated_subset', metrics['qd_scores'].get('validated', 0)),
            'QD Ratio': metrics['qd_scores']['ratio'],
            'QD Score (Full Archive)': metrics['qd_scores'].get('predicted_full_archive', 0),
            'N Compared': metrics.get('n_compared', metrics['n_valid']),
            'N Archive Total': metrics.get('n_archive_total', 0),
            'Coverage': metrics['archive_stats']['coverage'],
            'R²': metrics['metrics']['r2'],
            'RMSE': metrics['metrics']['rmse'],
            'MAE': metrics['metrics']['mae'],
            'Spearman ρ': metrics['metrics']['spearman_rho'],
            'Archive Size': metrics['n_valid'],
            'Mean Distance': diversity['mean_distance'],
            'SP Diversity': diversity['solow_polasky_diversity'],
            'Wall Time (min)': metrics.get('optimization_time', 0.0) / 60,
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_qd_score_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot QD score comparison across configurations."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = []
    qd_predicted = []
    qd_validated = []
    
    for config_name, result in results.items():
        metrics = result['metrics']
        configs.append(config_name)
        qd_predicted.append(metrics['qd_scores'].get('predicted_subset', metrics['qd_scores'].get('predicted', 0)))
        qd_validated.append(metrics['qd_scores'].get('validated_subset', metrics['qd_scores'].get('validated', 0)))
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, qd_predicted, width, label='Predicted', alpha=0.8)
    ax.bar(x + width/2, qd_validated, width, label='Validated (KLAM)', alpha=0.8)
    
    ax.set_ylabel('QD Score')
    ax.set_title('QD Score Comparison: Predicted vs Validated')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'qd_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'qd_score_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved QD score comparison plot")


def _parse_config_name(name: str) -> Tuple[str, float, int]:
    """Parse config name into (model, ucb_lambda, seed).
    
    Examples:
        'svgp_ucb1.0_size60_seed42' -> ('svgp', 1.0, 42)
        'unet_size60_seed42'        -> ('unet', 0.0, 42)
        'svgp_size60_seed42'        -> ('svgp', 0.0, 42)
    """
    m = re.match(r'^(\w+?)(?:_ucb([\d.]+))?_size\d+_seed(\d+)$', name)
    if m:
        model = m.group(1)
        ucb = float(m.group(2)) if m.group(2) else 0.0
        seed = int(m.group(3))
        return model, ucb, seed
    # Fallback
    return name, 0.0, 0


def plot_prediction_accuracy(results: Dict[str, Dict], output_dir: Path):
    """Plot prediction accuracy for each configuration.
    
    Subplots are arranged in a structured grid (3×8, horizontal):
        Rows  = replicate seeds (sorted ascending)
        Cols  = model × UCB-λ combinations (sorted: unet, svgp, hybrid)
    """
    
    # --- Parse & group configs ------------------------------------------------
    parsed = {name: _parse_config_name(name) for name in results}
    seeds = sorted(set(s for _, _, s in parsed.values()))
    
    # Build ordered column keys: (model, lambda)
    # Custom model order: unet first, then svgp, then hybrid
    model_order = {'unet': 0, 'svgp': 1, 'hybrid': 2}
    col_keys = sorted(
        set((m, l) for m, l, _ in parsed.values()),
        key=lambda x: (model_order.get(x[0], 99), x[1])
    )
    
    n_rows = len(seeds)
    n_cols = len(col_keys)
    
    # --- Create figure --------------------------------------------------------
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.5),
        squeeze=False
    )
    
    # Map (model, lambda, seed) -> config_name for quick lookup
    config_lookup = {v: k for k, v in parsed.items()}
    
    for row_idx, seed in enumerate(seeds):
        for col_idx, (model, ucb_lambda) in enumerate(col_keys):
            ax = axes[row_idx, col_idx]
            key = (model, ucb_lambda, seed)
            config_name = config_lookup.get(key)
            
            if config_name is None or config_name not in results:
                ax.set_visible(False)
                continue
            
            result = results[config_name]
            data = result['data']
            metrics_dict = result['metrics']
            
            # Filter valid solutions
            valid_mask = data['valid_mask']
            obj_pred = data['objectives_predicted'][valid_mask]
            obj_klam = data['objectives_klam'][valid_mask]
            
            # Scatter plot
            ax.scatter(obj_pred, obj_klam, alpha=0.5, s=10)
            
            # Perfect prediction line — use the combined range
            min_val = min(obj_pred.min(), obj_klam.min())
            max_val = max(obj_pred.max(), obj_klam.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', lw=2, label='Perfect')
            
            # Equal limits so diagonal is true 45°
            margin = (max_val - min_val) * 0.05
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            ax.set_aspect('equal', adjustable='box')
            
            # Metrics text
            r2 = metrics_dict['metrics']['r2']
            rho = metrics_dict['metrics']['spearman_rho']
            rmse = metrics_dict['metrics']['rmse']
            
            ax.text(0.05, 0.95,
                    f'R² = {r2:.3f}\nρ = {rho:.3f}\nRMSE = {rmse:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.tick_params(labelsize=14)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=14, loc='lower right')
            
            # --- Axis labels: only on edges ---
            if row_idx == n_rows - 1:
                ax.set_xlabel('Predicted Objective', fontsize=16)
            else:
                ax.set_xlabel('')
            if col_idx == 0:
                ax.set_ylabel('KLAM Objective\n(Ground Truth)', fontsize=16)
            else:
                ax.set_ylabel('')
            
            # --- Column headers (model×λ) on top row ---
            if row_idx == 0:
                label = model.upper()
                if ucb_lambda > 0:
                    label += f' λ={ucb_lambda}'
                ax.set_title(label, fontsize=16, fontweight='bold')
            else:
                ax.set_title('')
    
    # Row labels (seed) on the left side
    for row_idx, seed in enumerate(seeds):
        axes[row_idx, 0].annotate(
            f'Seed {seed}', xy=(0, 0.5),
            xytext=(-axes[row_idx, 0].yaxis.labelpad - 12, 0),
            xycoords='axes fraction', textcoords='offset points',
            fontsize=18, fontweight='bold',
            ha='right', va='center',
            rotation=90
        )
    
    plt.tight_layout(rect=(0.03, 0, 1, 1))  # Leave room for row labels
    plt.savefig(output_dir / 'prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'prediction_accuracy.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved prediction accuracy plot")


def plot_diversity_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot diversity comparison across configurations."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = []
    mean_distances = []
    sp_diversities = []
    
    for config_name, result in results.items():
        data = result['data']
        valid_mask = data['valid_mask']
        solutions = data['solutions'][valid_mask]
        
        diversity = compute_diversity_metrics(solutions)
        
        configs.append(config_name)
        mean_distances.append(diversity['mean_distance'])
        sp_diversities.append(diversity['solow_polasky_diversity'])
    
    # Mean pairwise distance
    ax1.bar(range(len(configs)), mean_distances, alpha=0.8)
    ax1.set_ylabel('Mean Pairwise Distance')
    ax1.set_title('Phenotypic Diversity: Mean Distance')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Solow-Polasky diversity
    ax2.bar(range(len(configs)), sp_diversities, alpha=0.8, color='orange')
    ax2.set_ylabel('Solow-Polasky Diversity')
    ax2.set_title('Phenotypic Diversity: SP Index')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'diversity_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved diversity comparison plot")


def plot_ucb_lambda_effect(results: Dict[str, Dict], output_dir: Path):
    """Plot effect of UCB lambda parameter."""
    
    # Group by model type
    by_model = {}
    for config_name, result in results.items():
        parts = config_name.split('_')
        model_type = parts[0]
        
        # Extract UCB lambda
        ucb_lambda = 0.0
        for part in parts:
            if part.startswith('ucb'):
                ucb_lambda = float(part.replace('ucb', ''))
        
        if model_type not in by_model:
            by_model[model_type] = []
        
        metrics = result['metrics']
        by_model[model_type].append({
            'lambda': ucb_lambda,
            'qd_validated': metrics['qd_scores'].get('validated_subset', metrics['qd_scores'].get('validated', 0)),
            'r2': metrics['metrics']['r2'],
            'coverage': metrics['archive_stats']['coverage'],
        })
    
    # Sort by lambda
    for model_type in by_model:
        by_model[model_type] = sorted(by_model[model_type], key=lambda x: x['lambda'])
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    for model_type, data_points in by_model.items():
        lambdas = [d['lambda'] for d in data_points]
        qd_scores = [d['qd_validated'] for d in data_points]
        r2_scores = [d['r2'] for d in data_points]
        coverages = [d['coverage'] for d in data_points]
        
        ax1.plot(lambdas, qd_scores, marker='o', label=model_type, linewidth=2)
        ax2.plot(lambdas, r2_scores, marker='o', label=model_type, linewidth=2)
        ax3.plot(lambdas, coverages, marker='o', label=model_type, linewidth=2)
    
    ax1.set_xlabel('UCB Lambda')
    ax1.set_ylabel('QD Score (Validated)')
    ax1.set_title('Effect of UCB λ on QD Score')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('UCB Lambda')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Effect of UCB λ on Prediction Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    ax3.set_xlabel('UCB Lambda')
    ax3.set_ylabel('Archive Coverage')
    ax3.set_title('Effect of UCB λ on Coverage')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ucb_lambda_effect.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ucb_lambda_effect.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved UCB lambda effect plot")


def analyze_results(results_dir: Path, output_dir: Path):
    """Main analysis function."""
    
    print("="*80)
    print("EXPERIMENT 6: QD COMPARISON ANALYSIS")
    print("="*80)
    
    # Load results
    print(f"\nLoading validation results from {results_dir}...")
    results = load_validation_results(results_dir)
    
    print(f"Found {len(results)} configurations:")
    for config_name in results.keys():
        print(f"  - {config_name}")
    
    if len(results) == 0:
        print("\nNo validation results found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    
    # Save table
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    
    # Print table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Generate plots
    print("\nGenerating plots...")
    plot_qd_score_comparison(results, output_dir)
    plot_prediction_accuracy(results, output_dir)
    plot_diversity_comparison(results, output_dir)
    
    # UCB lambda effect (if applicable)
    if any('ucb' in name for name in results.keys()):
        plot_ucb_lambda_effect(results, output_dir)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Best configuration by QD score
    best_qd_config = df.loc[df['QD Score (Validated)'].idxmax()]
    print(f"\nBest QD Score (Validated):")
    print(f"  Configuration: {best_qd_config['Configuration']}")
    print(f"  QD Score: {best_qd_config['QD Score (Validated)']:.2f}")
    print(f"  R²: {best_qd_config['R²']:.4f}")
    print(f"  Coverage: {best_qd_config['Coverage']:.2%}")
    
    # Best configuration by prediction accuracy
    best_r2_config = df.loc[df['R²'].idxmax()]
    print(f"\nBest Prediction Accuracy (R²):")
    print(f"  Configuration: {best_r2_config['Configuration']}")
    print(f"  R²: {best_r2_config['R²']:.4f}")
    print(f"  QD Score: {best_r2_config['QD Score (Validated)']:.2f}")
    
    # Best configuration by diversity
    best_div_config = df.loc[df['SP Diversity'].idxmax()]
    print(f"\nBest Diversity (SP Index):")
    print(f"  Configuration: {best_div_config['Configuration']}")
    print(f"  SP Diversity: {best_div_config['SP Diversity']:.2f}")
    print(f"  Mean Distance: {best_div_config['Mean Distance']:.4f}")
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 6 results")
    
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with validation results')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    analyze_results(Path(args.results_dir), Path(args.output_dir))


if __name__ == '__main__':
    main()
