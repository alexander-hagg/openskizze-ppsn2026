#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Flux Sensitivity Experiment: Analysis and Visualization

Analyzes results from the factorial experiment and generates:
1. Flux response heatmaps (GRZ × Height) for each orientation
2. Thumbnail grid showing all layouts with flux overlay
3. Correlation analysis between flux and morphological features
4. Statistical tests for factor significance

Usage:
    python experiments/flux_sensitivity/analyze_results.py
    python experiments/flux_sensitivity/analyze_results.py --data-dir results/flux_sensitivity
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(data_dir: str, debug: bool = False) -> Dict:
    """Load experiment results from NPZ files."""
    data_path = Path(data_dir) / "flux_sensitivity_results.npz"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Results not found: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    if debug:
        print(f"\n[DEBUG] NPZ file keys: {list(data.keys())}")
        for key in data.keys():
            arr = data[key]
            print(f"  {key}: shape={getattr(arr, 'shape', 'N/A')}, dtype={getattr(arr, 'dtype', type(arr).__name__)}")
    
    results = {
        'config_ids': data['config_ids'],
        'grz': data['grz'],
        'height_floors': data['height_floors'],
        'orientation_deg': data['orientation_deg'],
        'labels': data['labels'],
        'fitness': data['fitness'],
        'features': data['features'],
        'heightmaps': data['heightmaps'],
        'feature_names': data['feature_names'],
        'parcel_cells': int(data['parcel_cells']),
        'xy_scale': float(data['xy_scale']),
        'z_scale': float(data['z_scale']),
    }
    
    if debug:
        print(f"\n[DEBUG] Fitness values loaded:")
        print(f"  Shape: {results['fitness'].shape}")
        print(f"  Dtype: {results['fitness'].dtype}")
        print(f"  Min: {np.nanmin(results['fitness']):.4f}")
        print(f"  Max: {np.nanmax(results['fitness']):.4f}")
        print(f"  Mean: {np.nanmean(results['fitness']):.4f}")
        print(f"  NaN count: {np.sum(np.isnan(results['fitness']))}")
        print(f"  Unique values: {np.unique(results['fitness'][~np.isnan(results['fitness'])])[:20]}...")
        
        print(f"\n[DEBUG] Per-config breakdown:")
        for i, (label, grz, h, o, fit) in enumerate(zip(
            results['labels'], results['grz'], results['height_floors'],
            results['orientation_deg'], results['fitness']
        )):
            status = "VALID" if not np.isnan(fit) and fit != 1.0 else "FAIL?" if fit == 1.0 else "NaN"
            print(f"  [{i:2d}] {label}: GRZ={grz:.2f}, H={h}, O={o}° -> fitness={fit:.4f} [{status}]")
    
    # Try loading spatial data
    spatial_path = Path(data_dir) / "flux_sensitivity_spatial.npz"
    if spatial_path.exists():
        spatial = np.load(spatial_path, allow_pickle=True)
        results['uq'] = spatial['uq']
        results['vq'] = spatial['vq']
        results['spatial_indices'] = spatial['config_indices']
        
        if debug:
            print(f"\n[DEBUG] Spatial data loaded:")
            print(f"  uq shape: {results['uq'].shape if hasattr(results['uq'], 'shape') else 'N/A'}")
            print(f"  vq shape: {results['vq'].shape if hasattr(results['vq'], 'shape') else 'N/A'}")
            print(f"  config_indices: {results['spatial_indices']}")
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def create_flux_matrix(results: Dict, orientation: int) -> Tuple[np.ndarray, List, List]:
    """
    Create a GRZ × Height matrix of flux values for a specific orientation.
    
    Returns:
        matrix: 2D array with flux values
        grz_levels: List of GRZ values (columns)
        height_levels: List of height values (rows)
    """
    # Get unique levels
    grz_levels = sorted(set(results['grz']))
    height_levels = sorted(set(results['height_floors']))
    
    # Create matrix
    matrix = np.full((len(height_levels), len(grz_levels)), np.nan)
    
    # Fill matrix
    for i, (grz, height, orient, fitness) in enumerate(zip(
        results['grz'], results['height_floors'], 
        results['orientation_deg'], results['fitness']
    )):
        if orient == orientation:
            row = height_levels.index(height)
            col = grz_levels.index(grz)
            matrix[row, col] = fitness
    
    return matrix, grz_levels, height_levels


def compute_correlations(results: Dict) -> pd.DataFrame:
    """Compute correlations between flux and all features."""
    valid = ~np.isnan(results['fitness'])
    fitness = results['fitness'][valid]
    features = results['features'][valid]
    feature_names = list(results['feature_names'])
    
    correlations = []
    for i, name in enumerate(feature_names):
        feat = features[:, i]
        
        # Skip if no variation
        if np.std(feat) < 1e-10:
            correlations.append({
                'feature': name,
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_rho': np.nan,
                'spearman_p': np.nan
            })
            continue
        
        # Pearson correlation
        r, p_pearson = stats.pearsonr(feat, fitness)
        
        # Spearman rank correlation
        rho, p_spearman = stats.spearmanr(feat, fitness)
        
        correlations.append({
            'feature': name,
            'pearson_r': r,
            'pearson_p': p_pearson,
            'spearman_rho': rho,
            'spearman_p': p_spearman
        })
    
    return pd.DataFrame(correlations)


def compute_anova(results: Dict) -> Dict:
    """
    Compute factorial ANOVA to determine factor significance.
    
    Returns dict with F-statistics and p-values for each factor.
    """
    valid = ~np.isnan(results['fitness'])
    
    df = pd.DataFrame({
        'fitness': results['fitness'][valid],
        'grz': results['grz'][valid],
        'height': results['height_floors'][valid],
        'orientation': results['orientation_deg'][valid]
    })
    
    # Simple one-way ANOVAs for each factor
    anova_results = {}
    
    # GRZ effect
    groups_grz = [df[df['grz'] == g]['fitness'].values for g in df['grz'].unique()]
    groups_grz = [g for g in groups_grz if len(g) > 0]
    if len(groups_grz) > 1:
        f_grz, p_grz = stats.f_oneway(*groups_grz)
        anova_results['GRZ'] = {'F': f_grz, 'p': p_grz}
    
    # Height effect
    groups_h = [df[df['height'] == h]['fitness'].values for h in df['height'].unique()]
    groups_h = [g for g in groups_h if len(g) > 0]
    if len(groups_h) > 1:
        f_h, p_h = stats.f_oneway(*groups_h)
        anova_results['Height'] = {'F': f_h, 'p': p_h}
    
    # Orientation effect
    groups_o = [df[df['orientation'] == o]['fitness'].values for o in df['orientation'].unique()]
    groups_o = [g for g in groups_o if len(g) > 0]
    if len(groups_o) > 1:
        f_o, p_o = stats.f_oneway(*groups_o)
        anova_results['Orientation'] = {'F': f_o, 'p': p_o}
    
    return anova_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_flux_heatmaps(results: Dict, output_dir: str):
    """
    Create 3-panel heatmap showing flux response for each orientation.
    """
    orientations = sorted(set(results['orientation_deg']))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('Cold Air Flux Response Surface\n(Higher = Better Airflow)', 
                 fontsize=14, fontweight='bold')
    
    # Find global min/max for consistent colorbar
    vmin = np.nanmin(results['fitness'])
    vmax = np.nanmax(results['fitness'])
    
    # Use diverging colormap centered on median
    cmap = plt.cm.RdYlBu  # Red (low/bad) -> Yellow -> Blue (high/good)
    
    for ax, orient in zip(axes, orientations):
        matrix, grz_levels, height_levels = create_flux_matrix(results, orient)
        
        # Flip matrix so height increases upward
        matrix_flipped = matrix[::-1, :]
        height_labels = height_levels[::-1]
        
        im = ax.imshow(matrix_flipped, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add value annotations
        for i in range(len(height_labels)):
            for j in range(len(grz_levels)):
                val = matrix_flipped[i, j]
                if not np.isnan(val):
                    # Choose text color based on background
                    text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           fontsize=8, color=text_color)
        
        # Labels
        ax.set_xticks(range(len(grz_levels)))
        ax.set_xticklabels([f'{g:.0%}' for g in grz_levels])
        ax.set_yticks(range(len(height_labels)))
        ax.set_yticklabels([f'{h} fl' for h in height_labels])
        
        ax.set_xlabel('Site Coverage (GRZ)')
        ax.set_ylabel('Building Height')
        
        orient_labels = {0: '0° (Parallel to Wind)', 45: '45° (Diagonal)', 90: '90° (Perpendicular)'}
        ax.set_title(orient_labels.get(orient, f'{orient}°'))
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label='Cold Air Flux (100 W/m²)')
    
    output_path = Path(output_dir) / "flux_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layout_thumbnails(results: Dict, output_dir: str):
    """
    Create a grid of thumbnails showing all layouts with flux values.
    """
    grz_levels = sorted(set(results['grz']))
    height_levels = sorted(set(results['height_floors']))
    orientations = sorted(set(results['orientation_deg']))
    
    n_cols = len(grz_levels)
    n_rows = len(height_levels)
    n_orient = len(orientations)
    
    fig = plt.figure(figsize=(n_cols * 2 * n_orient + 2, n_rows * 2 + 1))
    
    # Create outer grid for orientations
    outer_grid = GridSpec(1, n_orient + 1, figure=fig, width_ratios=[1]*n_orient + [0.05],
                          wspace=0.1)
    
    # Get colormap limits
    vmin = np.nanmin(results['fitness'])
    vmax = np.nanmax(results['fitness'])
    cmap = plt.cm.RdYlBu
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    for o_idx, orient in enumerate(orientations):
        # Create inner grid for this orientation
        inner_grid = outer_grid[o_idx].subgridspec(n_rows + 1, n_cols, hspace=0.1, wspace=0.1,
                                                    height_ratios=[0.3] + [1]*n_rows)
        
        # Title row
        ax_title = fig.add_subplot(inner_grid[0, :])
        ax_title.text(0.5, 0.5, f'Orientation: {orient}°', ha='center', va='center',
                     fontsize=12, fontweight='bold')
        ax_title.axis('off')
        
        for h_idx, height in enumerate(reversed(height_levels)):
            for g_idx, grz in enumerate(grz_levels):
                ax = fig.add_subplot(inner_grid[h_idx + 1, g_idx])
                
                # Find this configuration
                mask = ((results['grz'] == grz) & 
                       (results['height_floors'] == height) & 
                       (results['orientation_deg'] == orient))
                
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                    heightmap = results['heightmaps'][idx]
                    fitness = results['fitness'][idx]
                    
                    # Plot heightmap
                    ax.imshow(heightmap, cmap='Greys', vmin=0, 
                             vmax=max(height_levels) if max(height_levels) > 0 else 1)
                    
                    # Add fitness as colored border
                    if not np.isnan(fitness):
                        color = cmap(norm(fitness))
                        for spine in ax.spines.values():
                            spine.set_edgecolor(color)
                            spine.set_linewidth(3)
                        ax.set_title(f'{fitness:.2f}', fontsize=8, pad=2)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=8)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Row/column labels
                if o_idx == 0 and g_idx == 0:
                    ax.set_ylabel(f'{height}fl', fontsize=8)
                if h_idx == len(height_levels) - 1:
                    ax.set_xlabel(f'{grz:.0%}', fontsize=8)
    
    # Colorbar
    cbar_ax = fig.add_subplot(outer_grid[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label='Cold Air Flux')
    
    output_path = Path(output_dir) / "layout_thumbnails.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_airflow_vectors(results: Dict, output_dir: str, max_plots: int = 12):
    """
    Plot airflow vector fields for selected configurations.
    """
    if 'uq' not in results:
        print("No spatial data available for airflow plots")
        return
    
    # Select a subset of configurations to plot
    indices = results['spatial_indices']
    n_plots = min(max_plots, len(indices))
    
    # Select evenly spaced indices
    selected = np.linspace(0, len(indices) - 1, n_plots, dtype=int)
    
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
                             constrained_layout=True)
    axes = axes.flatten()
    
    for ax_idx, sel_idx in enumerate(selected):
        ax = axes[ax_idx]
        
        config_idx = indices[sel_idx]
        uq = results['uq'][sel_idx]
        vq = results['vq'][sel_idx]
        label = results['labels'][config_idx]
        fitness = results['fitness'][config_idx]
        heightmap = results['heightmaps'][config_idx]
        
        # Calculate wind speed magnitude
        speed = np.sqrt(uq**2 + vq**2)
        
        # Plot speed as background
        im = ax.imshow(speed, cmap='Blues', alpha=0.7)
        
        # Calculate correct offsets for the KLAM domain
        # KLAM extends the domain 100% to the left (upwind) for inlet buffer
        # Domain shape: (env_cells_y, env_cells_x) where env_cells_x > env_cells_y
        # env_cells_y = env_cells_base (rows)
        # env_cells_x = env_cells_base + left_extension (columns)
        parcel_cells = results['parcel_cells']
        env_cells_y, env_cells_x = speed.shape
        
        # Original offset (before extension) = (env_cells_base - parcel_cells) // 2
        # env_cells_base = env_cells_y (since y is not extended)
        original_offset = (env_cells_y - parcel_cells) // 2
        left_extension = original_offset  # Domain was extended by this amount
        
        # Parcel position:
        # - Row offset (y): original_offset
        # - Column offset (x): original_offset + left_extension
        offset_y = original_offset
        offset_x = original_offset + left_extension
        
        # Create building mask in full domain
        full_heightmap = np.zeros(speed.shape)
        if offset_y >= 0 and offset_x >= 0:
            full_heightmap[offset_y:offset_y+parcel_cells, offset_x:offset_x+parcel_cells] = heightmap
        
        # Overlay buildings
        ax.contour(full_heightmap, levels=[0.5], colors='red', linewidths=2)
        
        # Quiver plot (downsampled)
        skip = max(1, speed.shape[0] // 20)
        y, x = np.mgrid[0:speed.shape[0]:skip, 0:speed.shape[1]:skip]
        ax.quiver(x, y, uq[::skip, ::skip], vq[::skip, ::skip], 
                 scale=5, alpha=0.6, color='black')
        
        ax.set_title(f'{label}\nFlux: {fitness:.2f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')
    
    fig.suptitle('Airflow Patterns (Blue = Speed, Red = Buildings, Arrows = Direction)',
                 fontsize=12)
    
    output_path = Path(output_dir) / "airflow_vectors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlations(correlations: pd.DataFrame, output_dir: str):
    """Plot correlation bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = range(len(correlations))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], correlations['pearson_r'], 
                   width, label='Pearson r', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], correlations['spearman_rho'], 
                   width, label='Spearman ρ', color='darkorange')
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Correlation with Cold Air Flux')
    ax.set_title('Feature Correlations with Objective')
    ax.set_xticks(x)
    ax.set_xticklabels(correlations['feature'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "feature_correlations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(results: Dict, correlations: pd.DataFrame, 
                   anova: Dict, output_dir: str):
    """Generate text report with analysis summary."""
    report_path = Path(output_dir) / "analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FLUX SENSITIVITY EXPERIMENT - ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Data summary
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        valid = ~np.isnan(results['fitness'])
        f.write(f"Total configurations: {len(results['fitness'])}\n")
        f.write(f"Successful evaluations: {np.sum(valid)}\n")
        f.write(f"Failed evaluations: {np.sum(~valid)}\n\n")
        
        # Fitness statistics
        f.write("COLD AIR FLUX STATISTICS\n")
        f.write("-" * 40 + "\n")
        fitness = results['fitness'][valid]
        f.write(f"Min:    {np.min(fitness):.4f}\n")
        f.write(f"Max:    {np.max(fitness):.4f}\n")
        f.write(f"Mean:   {np.mean(fitness):.4f}\n")
        f.write(f"Median: {np.median(fitness):.4f}\n")
        f.write(f"Std:    {np.std(fitness):.4f}\n\n")
        
        # Extremes
        f.write("EXTREME CONFIGURATIONS\n")
        f.write("-" * 40 + "\n")
        min_idx = np.nanargmin(results['fitness'])
        max_idx = np.nanargmax(results['fitness'])
        f.write(f"Lowest flux:  {results['labels'][min_idx]} = {results['fitness'][min_idx]:.4f}\n")
        f.write(f"Highest flux: {results['labels'][max_idx]} = {results['fitness'][max_idx]:.4f}\n\n")
        
        # ANOVA results
        f.write("FACTOR SIGNIFICANCE (One-Way ANOVA)\n")
        f.write("-" * 40 + "\n")
        for factor, stats_dict in anova.items():
            sig = "***" if stats_dict['p'] < 0.001 else "**" if stats_dict['p'] < 0.01 else "*" if stats_dict['p'] < 0.05 else ""
            f.write(f"{factor:15s}: F = {stats_dict['F']:8.2f}, p = {stats_dict['p']:.4e} {sig}\n")
        f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n\n")
        
        # Correlations
        f.write("FEATURE CORRELATIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Feature':<25} {'Pearson r':>10} {'Spearman ρ':>12}\n")
        f.write("-" * 50 + "\n")
        for _, row in correlations.iterrows():
            f.write(f"{row['feature']:<25} {row['pearson_r']:>10.3f} {row['spearman_rho']:>12.3f}\n")
        f.write("\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        
        # Find strongest correlations
        strongest = correlations.iloc[np.abs(correlations['spearman_rho']).argmax()]
        f.write(f"Strongest predictor: {strongest['feature']} (ρ = {strongest['spearman_rho']:.3f})\n")
        
        # Find most significant factor
        if anova:
            most_sig = min(anova.items(), key=lambda x: x[1]['p'])
            f.write(f"Most significant factor: {most_sig[0]} (F = {most_sig[1]['F']:.2f})\n")
        
    print(f"Saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(data_dir: str, output_dir: str = None, debug: bool = False):
    """Run complete analysis pipeline."""
    print("=" * 70)
    print("FLUX SENSITIVITY EXPERIMENT - ANALYSIS")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = data_dir
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    results = load_results(data_dir, debug=debug)
    
    n_configs = len(results['fitness'])
    n_valid = np.sum(~np.isnan(results['fitness']))
    n_error = np.sum(results['fitness'] == 1.0)  # Old error fallback value
    print(f"Loaded {n_configs} configurations ({n_valid} valid, {n_error} with fitness=1.0)")
    
    if debug:
        # Show fitness distribution
        print(f"\n[DEBUG] Fitness statistics:")
        valid_fitness = results['fitness'][~np.isnan(results['fitness'])]
        print(f"  Range: [{np.min(valid_fitness):.4f}, {np.max(valid_fitness):.4f}]")
        print(f"  Mean: {np.mean(valid_fitness):.4f}")
        print(f"  Std: {np.std(valid_fitness):.4f}")
        
        # Count unique fitness values
        unique_vals, counts = np.unique(valid_fitness, return_counts=True)
        print(f"  Unique values ({len(unique_vals)}):")
        for v, c in zip(unique_vals, counts):
            print(f"    {v:.4f}: {c} occurrences")
    
    # Compute statistics
    print("\nComputing statistics...")
    correlations = compute_correlations(results)
    anova = compute_anova(results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_flux_heatmaps(results, output_dir)
    plot_layout_thumbnails(results, output_dir)
    plot_correlations(correlations, output_dir)
    
    if 'uq' in results:
        plot_airflow_vectors(results, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, correlations, anova, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - flux_heatmaps.png")
    print("  - layout_thumbnails.png")
    print("  - feature_correlations.png")
    if 'uq' in results:
        print("  - airflow_vectors.png")
    print("  - analysis_report.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze flux sensitivity experiment results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/flux_sensitivity",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as data-dir)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing all loaded data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        debug=args.debug
    )
