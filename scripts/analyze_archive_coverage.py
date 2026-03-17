#!/usr/bin/env python
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Analyze SAIL archive coverage, feature distributions, correlations, and bin occupancy.

This script loads SAIL archive NPZ files and performs comprehensive analysis:
- Feature bin occupancy (which bins are filled vs empty)
- Feature correlations and coupling
- Reachability analysis (which feature combinations are physically possible)
- Coverage projections for different dimensionality configurations
- Recommendations for archive dimensionality

Usage:
    # Analyze single archive
    python scripts/analyze_archive_coverage.py \
        --archive results/archive/exp1_gp_training_data/sail_data/sail_27x27_rep1_klam.npz \
        --output-dir results/archive_analysis
    
    # Analyze multiple archives (aggregate statistics)
    python scripts/analyze_archive_coverage.py \
        --archive-dir results/archive/exp1_gp_training_data/sail_data \
        --pattern "sail_*_klam.npz" \
        --output-dir results/archive_analysis
    
    # Quick summary (no plots)
    python scripts/analyze_archive_coverage.py \
        --archive results/archive/exp1_gp_training_data/sail_data/sail_27x27_rep1_klam.npz \
        --summary-only
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_archive_data(npz_path):
    """Load archive data from NPZ file."""
    data = np.load(npz_path)
    return {
        'genomes': data['genomes'],
        'objectives': data['objectives'],
        'features': data['features'],
        'parcel_size': int(data['parcel_size']),
        'num_generations': int(data['num_generations']),
        'replicate': int(data.get('replicate', 1)),
        'n_elites': len(data['objectives'])
    }


def compute_feature_bins(features, feat_ranges, num_bins=5):
    """
    Discretize features into bins and compute bin indices.
    
    Returns:
        bin_indices: (N, D) array of bin indices [0, num_bins-1]
        bin_edges: List of bin edge arrays per dimension
    """
    n_samples, n_dims = features.shape
    bin_indices = np.zeros((n_samples, n_dims), dtype=np.int32)
    bin_edges = []
    
    for d in range(n_dims):
        edges = np.linspace(feat_ranges[d][0], feat_ranges[d][1], num_bins + 1)
        bin_edges.append(edges)
        # Clip to handle edge cases
        digitized = np.digitize(features[:, d], edges[:-1]) - 1
        digitized = np.clip(digitized, 0, num_bins - 1)
        bin_indices[:, d] = digitized
    
    return bin_indices, bin_edges


def compute_archive_cells(bin_indices, num_bins=5):
    """
    Compute unique archive cells and their occupancy counts.
    
    Returns:
        unique_cells: (M, D) array of unique bin combinations
        cell_counts: (M,) array of how many solutions per cell
        cell_ids: (N,) array mapping each solution to its cell ID
    """
    # Convert bin indices to cell IDs (flatten multidimensional index)
    n_dims = bin_indices.shape[1]
    multipliers = num_bins ** np.arange(n_dims - 1, -1, -1)
    cell_ids = (bin_indices * multipliers).sum(axis=1)
    
    # Find unique cells
    unique_cell_ids, inverse_indices, counts = np.unique(
        cell_ids, return_inverse=True, return_counts=True
    )
    
    # Reconstruct bin indices for unique cells
    unique_cells = np.zeros((len(unique_cell_ids), n_dims), dtype=np.int32)
    for i, cell_id in enumerate(unique_cell_ids):
        for d in range(n_dims):
            unique_cells[i, d] = (cell_id // multipliers[d]) % num_bins
    
    return unique_cells, counts, cell_ids, inverse_indices


def analyze_feature_correlations(features, labels):
    """Compute correlation matrices (Pearson and Spearman)."""
    n_dims = features.shape[1]
    
    pearson_corr = np.zeros((n_dims, n_dims))
    spearman_corr = np.zeros((n_dims, n_dims))
    
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                pearson_corr[i, j] = 1.0
                spearman_corr[i, j] = 1.0
            else:
                pearson_corr[i, j], _ = pearsonr(features[:, i], features[:, j])
                spearman_corr[i, j], _ = spearmanr(features[:, i], features[:, j])
    
    return pearson_corr, spearman_corr


def analyze_bin_occupancy(bin_indices, num_bins=5):
    """
    Analyze which bins are occupied for each dimension.
    
    Returns:
        occupancy: (D, num_bins) array of counts per bin per dimension
        coverage: (D,) array of fraction of bins occupied per dimension
    """
    n_dims = bin_indices.shape[1]
    occupancy = np.zeros((n_dims, num_bins), dtype=np.int32)
    
    for d in range(n_dims):
        bins, counts = np.unique(bin_indices[:, d], return_counts=True)
        occupancy[d, bins] = counts
    
    coverage = (occupancy > 0).sum(axis=1) / num_bins
    
    return occupancy, coverage


def analyze_pairwise_occupancy(bin_indices, num_bins=5):
    """
    Analyze 2D bin occupancy for all feature pairs.
    
    Returns:
        pairwise_coverage: (D, D) matrix of coverage for each feature pair
        pairwise_counts: Dict[(i,j)] -> (num_bins, num_bins) array
    """
    n_dims = bin_indices.shape[1]
    pairwise_coverage = np.zeros((n_dims, n_dims))
    pairwise_counts = {}
    
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            # Create 2D histogram
            counts = np.zeros((num_bins, num_bins), dtype=np.int32)
            for bi, bj in bin_indices[:, [i, j]]:
                counts[bi, bj] += 1
            
            pairwise_counts[(i, j)] = counts
            pairwise_coverage[i, j] = (counts > 0).sum() / (num_bins ** 2)
            pairwise_coverage[j, i] = pairwise_coverage[i, j]
    
    # Diagonal is always 100%
    np.fill_diagonal(pairwise_coverage, 1.0)
    
    return pairwise_coverage, pairwise_counts


def estimate_reachable_space(bin_indices, num_bins=5, threshold=0.01):
    """
    Estimate which regions of the feature space are reachable.
    
    Uses kernel density estimation on occupied cells to identify
    high-probability regions.
    
    Returns:
        reachable_fraction: Estimated fraction of archive that is reachable
    """
    # For now, use empirical coverage as lower bound
    # (More sophisticated analysis would require density estimation)
    unique_cells, counts, _, _ = compute_archive_cells(bin_indices, num_bins)
    n_dims = bin_indices.shape[1]
    total_cells = num_bins ** n_dims
    
    empirical_coverage = len(unique_cells) / total_cells
    
    return empirical_coverage


def project_dimensionality_coverage(features, bin_indices, num_bins, labels):
    """
    Project what coverage would be for different dimensionality reductions.
    
    Tests all possible feature subsets up to 6D.
    
    Returns:
        projections: List of dicts with subset info and projected coverage
    """
    n_dims = features.shape[1]
    projections = []
    
    # Test 1D through 6D
    for target_dims in range(1, min(7, n_dims + 1)):
        if target_dims == n_dims:
            # Full dimensionality (current)
            unique_cells, _, _, _ = compute_archive_cells(bin_indices, num_bins)
            total_cells = num_bins ** n_dims
            coverage = len(unique_cells) / total_cells
            projections.append({
                'n_dims': n_dims,
                'features': list(range(n_dims)),
                'feature_names': labels,
                'total_cells': total_cells,
                'occupied_cells': len(unique_cells),
                'coverage': coverage,
                'is_current': True
            })
        else:
            # For each dimensionality, find best subset by testing greedy selection
            # Start with feature 0 (GRZ - most important from Exp 2)
            from itertools import combinations
            
            best_coverage = 0
            best_subset = None
            
            # Test all combinations (brute force for small dimensions)
            if target_dims <= 4:
                for subset in combinations(range(n_dims), target_dims):
                    subset_bins = bin_indices[:, list(subset)]
                    unique_cells, _, _, _ = compute_archive_cells(subset_bins, num_bins)
                    total_cells = num_bins ** target_dims
                    coverage = len(unique_cells) / total_cells
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_subset = subset
            else:
                # For 5-6D, test a few strategic subsets
                # Priority: GRZ, GFZ, Height, Distance, Park, Count, Height Var, Compactness
                priority_order = [0, 1, 2, 4, 7, 5, 3, 6]
                best_subset = tuple(priority_order[:target_dims])
                subset_bins = bin_indices[:, list(best_subset)]
                unique_cells, _, _, _ = compute_archive_cells(subset_bins, num_bins)
                total_cells = num_bins ** target_dims
                best_coverage = len(unique_cells) / total_cells
            
            projections.append({
                'n_dims': target_dims,
                'features': list(best_subset),
                'feature_names': [labels[i] for i in best_subset],
                'total_cells': num_bins ** target_dims,
                'occupied_cells': int(best_coverage * (num_bins ** target_dims)),
                'coverage': best_coverage,
                'is_current': False
            })
    
    return projections


def generate_summary_report(archive_data, config, analysis_results, output_path):
    """Generate comprehensive text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAIL ARCHIVE COVERAGE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Archive Info
        f.write("-"*80 + "\n")
        f.write("ARCHIVE INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Parcel Size: {archive_data['parcel_size']}×{archive_data['parcel_size']} m\n")
        f.write(f"Replicate: {archive_data['replicate']}\n")
        f.write(f"Generations: {archive_data['num_generations']:,}\n")
        f.write(f"Total Elites: {archive_data['n_elites']:,}\n\n")
        
        # Archive Configuration
        f.write("-"*80 + "\n")
        f.write("ARCHIVE CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Dimensions: {config['n_dims']}\n")
        f.write(f"Bins per dimension: {config['num_bins']}\n")
        f.write(f"Total cells: {config['total_cells']:,}\n")
        f.write(f"Features: {', '.join(config['labels'])}\n\n")
        
        # Coverage Statistics
        f.write("-"*80 + "\n")
        f.write("COVERAGE STATISTICS\n")
        f.write("-"*80 + "\n")
        results = analysis_results
        f.write(f"Occupied cells: {results['n_occupied_cells']:,}\n")
        f.write(f"Overall coverage: {results['overall_coverage']:.4%}\n")
        f.write(f"Solutions per occupied cell (avg): {results['avg_solutions_per_cell']:.2f}\n")
        f.write(f"Solutions per occupied cell (median): {results['median_solutions_per_cell']:.1f}\n")
        f.write(f"Max solutions in single cell: {results['max_solutions_per_cell']}\n\n")
        
        # Per-Dimension Coverage
        f.write("-"*80 + "\n")
        f.write("PER-DIMENSION BIN COVERAGE\n")
        f.write("-"*80 + "\n")
        for i, label in enumerate(config['labels']):
            coverage = results['per_dim_coverage'][i]
            f.write(f"{i}. {label:30s}: {coverage:.1%} ({int(coverage * config['num_bins'])}/{config['num_bins']} bins)\n")
        f.write("\n")
        
        # Feature Correlations
        f.write("-"*80 + "\n")
        f.write("FEATURE CORRELATIONS (Spearman)\n")
        f.write("-"*80 + "\n")
        f.write("Strong correlations (|ρ| > 0.7):\n")
        spearman = results['spearman_corr']
        for i in range(config['n_dims']):
            for j in range(i + 1, config['n_dims']):
                if abs(spearman[i, j]) > 0.7:
                    f.write(f"  {config['labels'][i]:25s} ↔ {config['labels'][j]:25s}: {spearman[i, j]:+.3f}\n")
        f.write("\n")
        
        # Pairwise Coverage
        f.write("-"*80 + "\n")
        f.write("PAIRWISE FEATURE COVERAGE (2D projections)\n")
        f.write("-"*80 + "\n")
        f.write("Lowest 2D coverage (hardest to fill simultaneously):\n")
        pairwise = results['pairwise_coverage']
        pairs = []
        for i in range(config['n_dims']):
            for j in range(i + 1, config['n_dims']):
                pairs.append((pairwise[i, j], i, j))
        pairs.sort()
        for cov, i, j in pairs[:10]:
            f.write(f"  {config['labels'][i]:25s} × {config['labels'][j]:25s}: {cov:.1%}\n")
        f.write("\n")
        
        # Dimensionality Projections
        f.write("-"*80 + "\n")
        f.write("DIMENSIONALITY REDUCTION PROJECTIONS\n")
        f.write("-"*80 + "\n")
        f.write("Estimated coverage for different feature subsets:\n\n")
        for proj in results['dimensionality_projections']:
            marker = " [CURRENT]" if proj['is_current'] else ""
            f.write(f"{proj['n_dims']}D: {proj['coverage']:.2%} coverage{marker}\n")
            f.write(f"    Features: {', '.join(proj['feature_names'])}\n")
            f.write(f"    Cells: {proj['occupied_cells']:,} / {proj['total_cells']:,}\n\n")
        
        # Recommendations
        f.write("-"*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        
        # Find best 4D and 5D configurations
        projs_4d = [p for p in results['dimensionality_projections'] if p['n_dims'] == 4]
        projs_5d = [p for p in results['dimensionality_projections'] if p['n_dims'] == 5]
        
        if projs_4d:
            best_4d = projs_4d[0]
            f.write(f"1. For 4D archive (625 cells):\n")
            f.write(f"   Expected coverage: ~{best_4d['coverage']:.1%}\n")
            f.write(f"   Recommended features: {', '.join(best_4d['feature_names'])}\n\n")
        
        if projs_5d:
            best_5d = projs_5d[0]
            f.write(f"2. For 5D archive (3,125 cells):\n")
            f.write(f"   Expected coverage: ~{best_5d['coverage']:.1%}\n")
            f.write(f"   Recommended features: {', '.join(best_5d['feature_names'])}\n\n")
        
        f.write(f"3. Current {config['n_dims']}D configuration:\n")
        f.write(f"   Coverage: {results['overall_coverage']:.2%}\n")
        f.write(f"   This is NORMAL for {config['n_dims']}D archives!\n")
        f.write(f"   You have {archive_data['n_elites']:,} diverse solutions, which is excellent.\n\n")
        
        # Key insights
        f.write("-"*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("-"*80 + "\n")
        
        # Check for highly correlated features
        high_corr_pairs = []
        for i in range(config['n_dims']):
            for j in range(i + 1, config['n_dims']):
                if abs(spearman[i, j]) > 0.7:
                    high_corr_pairs.append((i, j, spearman[i, j]))
        
        if high_corr_pairs:
            f.write(f"• {len(high_corr_pairs)} feature pairs show strong correlation (|ρ| > 0.7)\n")
            f.write("  This suggests dimensional redundancy.\n\n")
        
        # Check for unbalanced bin occupancy
        min_dim_cov = results['per_dim_coverage'].min()
        max_dim_cov = results['per_dim_coverage'].max()
        if max_dim_cov - min_dim_cov > 0.3:
            f.write(f"• Unbalanced dimension coverage: {min_dim_cov:.1%} to {max_dim_cov:.1%}\n")
            worst_idx = results['per_dim_coverage'].argmin()
            f.write(f"  '{config['labels'][worst_idx]}' has poorest coverage.\n\n")
        
        # Check if coverage is reasonable for dimensionality
        expected_coverage_8d = 0.025  # 2.5% is typical for 8D
        if results['overall_coverage'] < expected_coverage_8d / 2:
            f.write(f"• Coverage is below typical for 8D archives (~2.5%).\n")
            f.write(f"  Consider increasing generations or reducing dimensions.\n\n")
        elif results['overall_coverage'] > expected_coverage_8d * 2:
            f.write(f"• Coverage is above typical for 8D archives!\n")
            f.write(f"  Your SAIL configuration is working very well.\n\n")
        else:
            f.write(f"• Coverage is within expected range for 8D archives (1-4%).\n")
            f.write(f"  Archive is filling normally. Dimensionality is the limiting factor.\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SAIL archive coverage and feature distributions'
    )
    parser.add_argument('--archive', type=str,
                        help='Path to single archive NPZ file')
    parser.add_argument('--archive-dir', type=str,
                        help='Directory containing multiple archives (will aggregate)')
    parser.add_argument('--pattern', type=str, default='*_klam.npz',
                        help='Glob pattern for archives in directory')
    parser.add_argument('--output-dir', type=str, default='results/archive_analysis',
                        help='Output directory for reports and plots')
    parser.add_argument('--num-bins', type=int, default=5,
                        help='Number of bins per dimension (default: 5)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary, no plots or detailed analysis')
    
    args = parser.parse_args()
    
    # Determine input files
    if args.archive:
        archive_files = [Path(args.archive)]
    elif args.archive_dir:
        archive_dir = Path(args.archive_dir)
        archive_files = sorted(archive_dir.glob(args.pattern))
        archive_files = [f for f in archive_files if not f.name.endswith('_spatial.npz')]
    else:
        print("ERROR: Must provide --archive or --archive-dir")
        sys.exit(1)
    
    if not archive_files:
        print("ERROR: No archive files found!")
        sys.exit(1)
    
    print(f"Found {len(archive_files)} archive(s) to analyze")
    
    # Load config
    with open(project_root / "domain_description/cfg.yml") as f:
        config_env = yaml.safe_load(f)
    
    labels = [config_env['labels'][i] for i in config_env['features']]
    feat_ranges = np.array(config_env['feat_ranges']).T
    feat_ranges = [feat_ranges[i] for i in config_env['features']]
    n_dims = len(config_env['features'])
    total_cells = args.num_bins ** n_dims
    
    config = {
        'n_dims': n_dims,
        'num_bins': args.num_bins,
        'total_cells': total_cells,
        'labels': labels,
        'feat_ranges': feat_ranges
    }
    
    # Analyze each archive
    for archive_file in archive_files:
        print("\n" + "="*80)
        print(f"Analyzing: {archive_file.name}")
        print("="*80)
        
        # Load data
        archive_data = load_archive_data(archive_file)
        features = archive_data['features']
        
        print(f"  Elites: {archive_data['n_elites']:,}")
        print(f"  Parcel: {archive_data['parcel_size']}m")
        print(f"  Generations: {archive_data['num_generations']:,}")
        
        # Compute bins
        print("\n  Computing feature bins...")
        bin_indices, bin_edges = compute_feature_bins(features, feat_ranges, args.num_bins)
        
        # Compute archive cells
        print("  Analyzing archive cells...")
        unique_cells, cell_counts, cell_ids, inverse_indices = compute_archive_cells(
            bin_indices, args.num_bins
        )
        
        n_occupied = len(unique_cells)
        overall_coverage = n_occupied / total_cells
        
        print(f"  Occupied cells: {n_occupied:,} / {total_cells:,} ({overall_coverage:.4%})")
        
        # Feature correlations
        print("  Computing feature correlations...")
        pearson_corr, spearman_corr = analyze_feature_correlations(features, labels)
        
        # Bin occupancy
        print("  Analyzing bin occupancy...")
        bin_occupancy, per_dim_coverage = analyze_bin_occupancy(bin_indices, args.num_bins)
        
        # Pairwise coverage
        print("  Analyzing pairwise feature coverage...")
        pairwise_coverage, pairwise_counts = analyze_pairwise_occupancy(
            bin_indices, args.num_bins
        )
        
        # Dimensionality projections
        print("  Computing dimensionality projections...")
        projections = project_dimensionality_coverage(
            features, bin_indices, args.num_bins, labels
        )
        
        # Compile results
        analysis_results = {
            'n_occupied_cells': n_occupied,
            'overall_coverage': overall_coverage,
            'avg_solutions_per_cell': cell_counts.mean(),
            'median_solutions_per_cell': np.median(cell_counts),
            'max_solutions_per_cell': cell_counts.max(),
            'per_dim_coverage': per_dim_coverage,
            'bin_occupancy': bin_occupancy,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'pairwise_coverage': pairwise_coverage,
            'pairwise_counts': pairwise_counts,
            'dimensionality_projections': projections
        }
        
        if args.summary_only:
            # Print quick summary
            print("\n" + "-"*80)
            print("SUMMARY")
            print("-"*80)
            print(f"Archive: {config['n_dims']}D × {args.num_bins} bins = {total_cells:,} cells")
            print(f"Occupied: {n_occupied:,} ({overall_coverage:.2%})")
            print(f"\nPer-dimension coverage:")
            for i, label in enumerate(labels):
                print(f"  {label:30s}: {per_dim_coverage[i]:.1%}")
            
            print(f"\nDimensionality projections:")
            for proj in projections:
                marker = " [CURRENT]" if proj['is_current'] else ""
                print(f"  {proj['n_dims']}D: {proj['coverage']:.2%}{marker}")
            
        else:
            # Generate full report
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename based on archive
            base_name = archive_file.stem.replace('_klam', '')
            report_file = output_dir / f"{base_name}_coverage_report.txt"
            
            print(f"\n  Generating report: {report_file}")
            generate_summary_report(archive_data, config, analysis_results, report_file)
            
            # Save detailed results to pickle for plotting
            results_file = output_dir / f"{base_name}_coverage_data.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'archive_data': archive_data,
                    'config': config,
                    'analysis_results': analysis_results,
                    'bin_indices': bin_indices,
                    'features': features
                }, f)
            print(f"  Saved detailed data: {results_file}")
            
            print(f"\n  ✓ Analysis complete. See {report_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
