#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Analyze MAP-Elites GP Experiment Results

This script loads all MAP-Elites runs from the grid sweep and generates
comprehensive analysis comparing QD metrics and diversity across configurations.

Usage:
    python experiments/analyze_mapelites_gp.py --results-dir results/mapelites_gp
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.
    
    Returns:
        DataFrame with columns: num_emitters, batch_size, replicate, generation,
                                 qd_score, coverage, max_objective, mean_distance,
                                 sp_diversity, effective_n, wall_time
    """
    rows = []
    
    for run_dir in sorted(results_dir.glob("emit*_batch*_rep*")):
        # Parse directory name
        parts = run_dir.name.split('_')
        num_emitters = int(parts[0].replace('emit', ''))
        batch_size = int(parts[1].replace('batch', ''))
        replicate = int(parts[2].replace('rep', ''))
        
        # Load history
        history_path = run_dir / "history.pkl"
        if not history_path.exists():
            logger.warning(f"Missing history: {history_path}")
            continue
        
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        # Extract data for each generation
        for gen_idx, gen in enumerate(history['generations']):
            # Extract diversity metrics for this generation
            div_metrics = history['diversity_metrics'][gen_idx]
            
            row = {
                'num_emitters': num_emitters,
                'batch_size': batch_size,
                'replicate': replicate,
                'generation': gen,
                'qd_score': history['qd_score'][gen_idx],
                'coverage': history['coverage'][gen_idx],
                'max_objective': history['max_objective'][gen_idx],
                'mean_distance': div_metrics['mean_pairwise_distance'],
                'sp_diversity': div_metrics['solow_polasky_diversity'],
                'effective_n': div_metrics['effective_n_species'],
                'wall_time': history['wall_time'][gen_idx],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} data points from {len(df.groupby(['num_emitters', 'batch_size', 'replicate']))} runs")
    
    return df


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary stats at final generation."""
    # Get final generation for each run
    df_final = df.groupby(['num_emitters', 'batch_size', 'replicate']).last().reset_index()
    
    # Aggregate across replicates
    summary = df_final.groupby(['num_emitters', 'batch_size']).agg({
        'qd_score': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'max_objective': ['mean', 'std'],
        'mean_distance': ['mean', 'std'],
        'sp_diversity': ['mean', 'std'],
        'effective_n': ['mean', 'std'],
        'wall_time': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    return summary


def plot_evolution_curves(df: pd.DataFrame, output_dir: Path):
    """Plot QD metrics over generations."""
    metrics = ['qd_score', 'coverage', 'max_objective']
    titles = ['QD Score', 'Coverage (%)', 'Max Objective']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric, title in zip(axes, metrics, titles):
        for num_emitters in sorted(df['num_emitters'].unique()):
            for batch_size in sorted(df['batch_size'].unique()):
                subset = df[(df['num_emitters'] == num_emitters) & 
                           (df['batch_size'] == batch_size)]
                
                if len(subset) == 0:
                    continue
                
                # Aggregate across replicates
                grouped = subset.groupby('generation')[metric].agg(['mean', 'std'])
                
                label = f"E={num_emitters}, B={batch_size}"
                ax.plot(grouped.index, grouped['mean'], label=label, alpha=0.8, linewidth=2)
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    alpha=0.2
                )
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evolution_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'evolution_curves.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved evolution curves to {output_dir}")


def plot_diversity_evolution(df: pd.DataFrame, output_dir: Path):
    """Plot diversity metrics over generations."""
    metrics = ['mean_distance', 'sp_diversity', 'effective_n']
    titles = ['Mean Pairwise Distance', 'Solow-Polasky Diversity', 'Effective N Species']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric, title in zip(axes, metrics, titles):
        for num_emitters in sorted(df['num_emitters'].unique()):
            for batch_size in sorted(df['batch_size'].unique()):
                subset = df[(df['num_emitters'] == num_emitters) & 
                           (df['batch_size'] == batch_size)]
                
                if len(subset) == 0:
                    continue
                
                # Aggregate across replicates
                grouped = subset.groupby('generation')[metric].agg(['mean', 'std'])
                
                label = f"E={num_emitters}, B={batch_size}"
                ax.plot(grouped.index, grouped['mean'], label=label, alpha=0.8, linewidth=2)
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    alpha=0.2
                )
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_evolution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'diversity_evolution.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved diversity evolution to {output_dir}")


def plot_final_performance_heatmaps(summary: pd.DataFrame, output_dir: Path):
    """Plot heatmaps of final performance across configurations."""
    metrics = [
        ('qd_score_mean', 'QD Score'),
        ('coverage_mean', 'Coverage (%)'),
        ('max_objective_mean', 'Max Objective'),
        ('mean_distance_mean', 'Mean Distance'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, (metric, title) in zip(axes, metrics):
        # Pivot for heatmap
        pivot = summary.pivot(
            index='num_emitters',
            columns='batch_size',
            values=metric
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': title}
        )
        ax.set_title(title)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Number of Emitters')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_performance_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'final_performance_heatmaps.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance heatmaps to {output_dir}")


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Plot Pareto frontier: QD score vs mean distance."""
    # Get final generation
    df_final = df.groupby(['num_emitters', 'batch_size', 'replicate']).last().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color by emitters, marker by batch size
    colors = {8: 'blue', 64: 'green', 256: 'red'}
    markers = {4: 'o', 16: 's', 64: '^', 128: 'D'}
    
    for num_emitters in sorted(df_final['num_emitters'].unique()):
        num_emitters = int(num_emitters)  # Convert from numpy int to Python int
        for batch_size in sorted(df_final['batch_size'].unique()):
            batch_size = int(batch_size)  # Convert from numpy int to Python int
            subset = df_final[
                (df_final['num_emitters'] == num_emitters) &
                (df_final['batch_size'] == batch_size)
            ]
            
            if len(subset) == 0:
                continue
            
            ax.scatter(
                subset['mean_distance'],
                subset['qd_score'],
                c=colors[num_emitters],
                marker=markers[batch_size],
                s=100,
                alpha=0.7,
                label=f"E={num_emitters}, B={batch_size}"
            )
    
    ax.set_xlabel('Mean Pairwise Distance (Phenotypic Diversity)')
    ax.set_ylabel('QD Score')
    ax.set_title('Pareto Frontier: QD Score vs Diversity')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_frontier.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Pareto frontier to {output_dir}")


def plot_efficiency_analysis(df: pd.DataFrame, output_dir: Path):
    """Plot QD score vs wall time (efficiency)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {8: 'blue', 64: 'green', 256: 'red'}
    markers = {4: 'o', 16: 's', 64: '^', 128: 'D'}
    
    for num_emitters in sorted(df['num_emitters'].unique()):
        num_emitters = int(num_emitters)  # Convert from numpy int to Python int
        for batch_size in sorted(df['batch_size'].unique()):
            batch_size = int(batch_size)  # Convert from numpy int to Python int
            subset = df[
                (df['num_emitters'] == num_emitters) &
                (df['batch_size'] == batch_size)
            ]
            
            if len(subset) == 0:
                continue
            
            # Aggregate across replicates
            grouped = subset.groupby('generation').agg({
                'wall_time': 'mean',
                'qd_score': 'mean'
            })
            
            ax.plot(
                grouped['wall_time'],
                grouped['qd_score'],
                color=colors[num_emitters],
                marker=markers[batch_size],
                alpha=0.7,
                linewidth=2,
                markersize=6,
                label=f"E={num_emitters}, B={batch_size}"
            )
    
    ax.set_xlabel('Wall Time (seconds)')
    ax.set_ylabel('QD Score')
    ax.set_title('Efficiency: QD Score vs Wall Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'efficiency_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved efficiency analysis to {output_dir}")


def perform_statistical_tests(df: pd.DataFrame, output_dir: Path):
    """Perform ANOVA and pairwise comparisons."""
    # Get final generation
    df_final = df.groupby(['num_emitters', 'batch_size', 'replicate']).last().reset_index()
    
    # Create configuration labels
    df_final['config'] = df_final.apply(
        lambda row: f"E{row['num_emitters']}_B{row['batch_size']}",
        axis=1
    )
    
    results = {}
    
    for metric in ['qd_score', 'coverage', 'max_objective', 'mean_distance']:
        logger.info(f"\nStatistical tests for {metric}:")
        
        # One-way ANOVA
        groups = [group[metric].values for name, group in df_final.groupby('config')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        logger.info(f"  ANOVA: F={f_stat:.2f}, p={p_value:.2e}")
        
        results[metric] = {
            'anova_f': float(f_stat),
            'anova_p': float(p_value)
        }
    
    # Save results
    import yaml
    with open(output_dir / 'statistical_tests.yaml', 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"\nSaved statistical tests to {output_dir / 'statistical_tests.yaml'}")


def generate_summary_report(summary: pd.DataFrame, output_dir: Path):
    """Generate markdown summary report."""
    report = ["# MAP-Elites GP Experiment Results", ""]
    
    report.append("## Final Performance Summary")
    report.append("")
    report.append("### QD Score")
    report.append("")
    qd_pivot = summary.pivot(
        index='num_emitters',
        columns='batch_size',
        values='qd_score_mean'
    )
    report.append(qd_pivot.to_markdown())
    report.append("")
    
    report.append("### Coverage (%)")
    report.append("")
    cov_pivot = summary.pivot(
        index='num_emitters',
        columns='batch_size',
        values='coverage_mean'
    )
    report.append(cov_pivot.to_markdown())
    report.append("")
    
    report.append("### Mean Pairwise Distance (Diversity)")
    report.append("")
    div_pivot = summary.pivot(
        index='num_emitters',
        columns='batch_size',
        values='mean_distance_mean'
    )
    report.append(div_pivot.to_markdown())
    report.append("")
    
    # Best configurations
    report.append("## Best Configurations")
    report.append("")
    
    best_qd = summary.loc[summary['qd_score_mean'].idxmax()]
    report.append(f"**Best QD Score**: E={best_qd['num_emitters']}, B={best_qd['batch_size']} "
                  f"({best_qd['qd_score_mean']:.2f} ± {best_qd['qd_score_std']:.2f})")
    report.append("")
    
    best_div = summary.loc[summary['mean_distance_mean'].idxmax()]
    report.append(f"**Best Diversity**: E={best_div['num_emitters']}, B={best_div['batch_size']} "
                  f"({best_div['mean_distance_mean']:.2f} ± {best_div['mean_distance_std']:.2f})")
    report.append("")
    
    # Write report
    with open(output_dir / 'SUMMARY.md', 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Generated summary report: {output_dir / 'SUMMARY.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MAP-Elites GP experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/mapelites_gp",
        help="Directory containing all experiment runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results-dir/analysis)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    if args.output_dir is None:
        output_dir = results_dir / "analysis"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MAP-Elites GP Experiment Analysis")
    logger.info("=" * 70)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("")
    
    # Load data
    logger.info("Loading all results...")
    df = load_all_results(results_dir)
    
    if len(df) == 0:
        logger.error("No results found!")
        return
    
    # Compute summary
    logger.info("\nComputing summary statistics...")
    summary = compute_summary_statistics(df)
    summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    logger.info(f"Saved summary to {output_dir / 'summary_statistics.csv'}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_evolution_curves(df, output_dir)
    plot_diversity_evolution(df, output_dir)
    plot_final_performance_heatmaps(summary, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_efficiency_analysis(df, output_dir)
    
    # Statistical tests
    logger.info("\nPerforming statistical tests...")
    perform_statistical_tests(df, output_dir)
    
    # Generate report
    logger.info("\nGenerating summary report...")
    generate_summary_report(summary, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Analysis complete!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
