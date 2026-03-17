#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Phase 2: Prepare Training Datasets

This script combines raw data from SAIL and Sobol generation into three balanced datasets:
1. Optimized (SAIL): Elite solutions from SAIL optimization
2. Random (Sobol): Random samples evaluated with KLAM_21  
3. Combined: 50% Optimized + 50% Random

Key steps:
- Pool samples from all parcel sizes and replicates
- Balance datasets to the minimum available size
- Split into train/validation (80/20)
- Save preprocessed datasets ready for GP training

Usage:
    python experiments/prepare_training_datasets.py --sail-dir results/sail_data --random-dir results/random_data
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Parcel sizes from experiment plan
PARCEL_SIZES = [60, 120, 240]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare balanced training datasets for GP experiment"
    )
    parser.add_argument(
        "--sail-dir",
        type=str,
        default="results/sail_data",
        help="Directory containing SAIL NPZ files"
    )
    parser.add_argument(
        "--random-dir",
        type=str,
        default="results/random_data",
        help="Directory containing random Sobol NPZ files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training_datasets",
        help="Output directory for prepared datasets"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Target number of samples per dataset (default: minimum available)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
    )
    return parser.parse_args()


def load_sail_data(sail_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load and pool all SAIL data files.
    
    Loads *_klam.npz files which contain real KLAM_21 objectives.
    Falls back to *.npz if no _klam.npz files exist.
    
    Returns dict with keys: genomes, widths, heights, objectives, features
    """
    # Prefer _klam.npz files (real KLAM_21 objectives)
    sail_files = list(sail_dir.glob("sail_*_klam.npz"))
    
    if sail_files:
        logger.info(f"Found {len(sail_files)} SAIL _klam.npz files (real KLAM_21 objectives)")
    else:
        # Fall back to old format
        sail_files = list(sail_dir.glob("sail_*.npz"))
        # Exclude _klam.npz and _spatial.npz
        sail_files = [f for f in sail_files if '_klam.npz' not in f.name and '_spatial.npz' not in f.name]
        if sail_files:
            logger.warning(f"No _klam.npz files found, using {len(sail_files)} old format files (surrogate objectives)")
    
    if not sail_files:
        raise FileNotFoundError(f"No SAIL data files found in {sail_dir}")
    
    logger.info(f"Found {len(sail_files)} SAIL data files")
    
    # Collect samples from all files
    all_genomes = []
    all_widths = []
    all_heights = []
    all_objectives = []
    all_features = []
    all_parcel_sizes = []
    
    for f in sorted(sail_files):
        data = np.load(f)
        n = len(data['genomes'])
        
        all_genomes.append(data['genomes'])
        all_widths.append(data['widths'])
        all_heights.append(data['heights'])
        all_objectives.append(data['objectives'])
        all_features.append(data['features'])
        
        # Handle both key names (parcel_size from SAIL, parcel_size_m from random)
        if 'parcel_size' in data:
            parcel_size = int(data['parcel_size'])
        elif 'parcel_size_m' in data:
            parcel_size = int(data['parcel_size_m'])
        else:
            # Try to extract from filename: sail_51x51_rep1_klam.npz
            import re
            match = re.search(r'sail_(\d+)x\d+', f.name)
            parcel_size = int(match.group(1)) if match else 0
        all_parcel_sizes.extend([parcel_size] * n)
        
        logger.info(f"  {f.name}: {n} samples, "
                    f"obj range [{data['objectives'].min():.2f}, {data['objectives'].max():.2f}]")
    
    result = {
        'genomes': np.vstack(all_genomes),
        'widths': np.concatenate(all_widths),
        'heights': np.concatenate(all_heights),
        'objectives': np.concatenate(all_objectives),
        'features': np.vstack(all_features),
        'parcel_sizes': np.array(all_parcel_sizes)
    }
    
    # Filter out failed evaluations (NaN objectives)
    valid_mask = ~np.isnan(result['objectives'])
    n_total = len(result['objectives'])
    n_valid = valid_mask.sum()
    
    if n_valid < n_total:
        logger.warning(f"Filtering out {n_total - n_valid} samples with NaN objectives")
        result = {k: v[valid_mask] for k, v in result.items()}
    
    logger.info(f"Total SAIL samples: {len(result['objectives'])} (after NaN filtering)")
    return result


def load_random_data(random_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load and pool all random Sobol data files.
    
    Returns dict with keys: genomes, widths, heights, objectives, features
    """
    random_files = list(random_dir.glob("random_*.npz"))
    # Exclude spatial data files
    random_files = [f for f in random_files if '_spatial.npz' not in f.name]
    
    if not random_files:
        raise FileNotFoundError(f"No random data files found in {random_dir}")
    
    logger.info(f"Found {len(random_files)} random data files")
    
    # Collect samples from all files
    all_genomes = []
    all_widths = []
    all_heights = []
    all_objectives = []
    all_features = []
    all_parcel_sizes = []
    
    for f in sorted(random_files):
        data = np.load(f)
        n = len(data['genomes'])
        
        all_genomes.append(data['genomes'])
        all_widths.append(data['widths'])
        all_heights.append(data['heights'])
        all_objectives.append(data['objectives'])
        all_features.append(data['features'])
        
        # Handle both key names (parcel_size from SAIL, parcel_size_m from random)
        if 'parcel_size_m' in data:
            parcel_size = int(data['parcel_size_m'])
        elif 'parcel_size' in data:
            parcel_size = int(data['parcel_size'])
        else:
            # Try to extract from filename: random_sobol_51m_n2000_seed42.npz
            import re
            match = re.search(r'random_sobol_(\d+)m', f.name)
            parcel_size = int(match.group(1)) if match else 0
        all_parcel_sizes.extend([parcel_size] * n)
        
        logger.info(f"  {f.name}: {n} samples, "
                    f"obj range [{data['objectives'].min():.2f}, {data['objectives'].max():.2f}]")
    
    result = {
        'genomes': np.vstack(all_genomes),
        'widths': np.concatenate(all_widths),
        'heights': np.concatenate(all_heights),
        'objectives': np.concatenate(all_objectives),
        'features': np.vstack(all_features),
        'parcel_sizes': np.array(all_parcel_sizes)
    }
    
    # Filter out failed evaluations (NaN objectives)
    valid_mask = ~np.isnan(result['objectives'])
    n_total = len(result['objectives'])
    n_valid = valid_mask.sum()
    
    if n_valid < n_total:
        logger.warning(f"Filtering out {n_total - n_valid} samples with NaN objectives")
        result = {k: v[valid_mask] for k, v in result.items()}
    
    logger.info(f"Total random samples: {len(result['objectives'])} (after NaN filtering)")
    return result


def subsample_data(
    data: Dict[str, np.ndarray],
    n_samples: int,
    seed: int
) -> Dict[str, np.ndarray]:
    """
    Randomly subsample data to n_samples.
    """
    rng = np.random.default_rng(seed)
    n_total = len(data['objectives'])
    
    if n_samples >= n_total:
        return data
    
    indices = rng.choice(n_total, size=n_samples, replace=False)
    
    return {
        'genomes': data['genomes'][indices],
        'widths': data['widths'][indices],
        'heights': data['heights'][indices],
        'objectives': data['objectives'][indices],
        'features': data['features'][indices],
        'parcel_sizes': data['parcel_sizes'][indices]
    }


def create_combined_dataset(
    sail_data: Dict[str, np.ndarray],
    random_data: Dict[str, np.ndarray],
    n_samples: int,
    seed: int
) -> Dict[str, np.ndarray]:
    """
    Create combined dataset with 50% optimized + 50% random.
    """
    n_each = n_samples // 2
    
    # Subsample each source
    sail_sub = subsample_data(sail_data, n_each, seed)
    random_sub = subsample_data(random_data, n_each, seed + 1)
    
    # Create source labels
    sail_sources = np.zeros(n_each, dtype=np.int32)  # 0 = SAIL
    random_sources = np.ones(n_each, dtype=np.int32)  # 1 = Random
    
    return {
        'genomes': np.vstack([sail_sub['genomes'], random_sub['genomes']]),
        'widths': np.concatenate([sail_sub['widths'], random_sub['widths']]),
        'heights': np.concatenate([sail_sub['heights'], random_sub['heights']]),
        'objectives': np.concatenate([sail_sub['objectives'], random_sub['objectives']]),
        'features': np.vstack([sail_sub['features'], random_sub['features']]),
        'parcel_sizes': np.concatenate([sail_sub['parcel_sizes'], random_sub['parcel_sizes']]),
        'source': np.concatenate([sail_sources, random_sources])
    }


def split_train_val(
    data: Dict[str, np.ndarray],
    val_fraction: float,
    seed: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into train and validation sets.
    """
    n_total = len(data['objectives'])
    indices = np.arange(n_total)
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed
    )
    
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    
    return train_data, val_data


def save_dataset(
    output_path: Path,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    metadata: Dict
):
    """
    Save dataset with train/val split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ with all data
    np.savez(
        output_path,
        # Training data
        train_genomes=train_data['genomes'],
        train_widths=train_data['widths'],
        train_heights=train_data['heights'],
        train_objectives=train_data['objectives'],
        train_features=train_data['features'],
        train_parcel_sizes=train_data['parcel_sizes'],
        # Validation data
        val_genomes=val_data['genomes'],
        val_widths=val_data['widths'],
        val_heights=val_data['heights'],
        val_objectives=val_data['objectives'],
        val_features=val_data['features'],
        val_parcel_sizes=val_data['parcel_sizes'],
        # Optional source labels for combined dataset
        train_source=train_data.get('source', np.array([])),
        val_source=val_data.get('source', np.array([])),
        # Metadata
        **{f"meta_{k}": v for k, v in metadata.items()}
    )
    
    logger.info(f"Saved dataset to {output_path}")
    logger.info(f"  Train: {len(train_data['objectives'])} samples")
    logger.info(f"  Val: {len(val_data['objectives'])} samples")


def compute_dataset_statistics(data: Dict[str, np.ndarray]) -> Dict:
    """
    Compute summary statistics for a dataset.
    """
    objectives = data['objectives']
    features = data['features']
    
    stats = {
        'n_samples': len(objectives),
        'objective_mean': float(objectives.mean()),
        'objective_std': float(objectives.std()),
        'objective_min': float(objectives.min()),
        'objective_max': float(objectives.max()),
        'feature_means': features.mean(axis=0).tolist(),
        'feature_stds': features.std(axis=0).tolist(),
    }
    
    # Parcel size distribution
    parcel_sizes = data['parcel_sizes']
    unique, counts = np.unique(parcel_sizes, return_counts=True)
    stats['parcel_size_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
    
    return stats


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 2: Prepare Training Datasets")
    logger.info("=" * 60)
    
    sail_dir = Path(args.sail_dir)
    random_dir = Path(args.random_dir)
    output_dir = Path(args.output_dir)
    
    # Load all raw data
    logger.info("\nLoading SAIL data...")
    sail_data = load_sail_data(sail_dir)
    
    logger.info("\nLoading random data...")
    random_data = load_random_data(random_dir)
    
    # Determine target sample size
    n_sail = len(sail_data['objectives'])
    n_random = len(random_data['objectives'])
    
    if args.target_samples:
        n_target = args.target_samples
    else:
        # Use minimum of available samples
        n_target = min(n_sail, n_random)
    
    logger.info(f"\nTarget samples per dataset: {n_target}")
    logger.info(f"  SAIL available: {n_sail}")
    logger.info(f"  Random available: {n_random}")
    
    if n_target > n_sail:
        logger.warning(f"Target ({n_target}) exceeds SAIL samples ({n_sail})")
        n_target = n_sail
    if n_target > n_random:
        logger.warning(f"Target ({n_target}) exceeds random samples ({n_random})")
        n_target = n_random
    
    # Create three datasets
    datasets = {}
    
    # 1. Optimized (SAIL only)
    logger.info("\n--- Creating OPTIMIZED dataset (SAIL) ---")
    opt_data = subsample_data(sail_data, n_target, args.seed)
    opt_train, opt_val = split_train_val(opt_data, args.val_fraction, args.seed)
    datasets['optimized'] = {'train': opt_train, 'val': opt_val}
    
    # 2. Random (Sobol only)
    logger.info("\n--- Creating RANDOM dataset (Sobol) ---")
    rand_data = subsample_data(random_data, n_target, args.seed)
    rand_train, rand_val = split_train_val(rand_data, args.val_fraction, args.seed)
    datasets['random'] = {'train': rand_train, 'val': rand_val}
    
    # 3. Combined (50% each)
    logger.info("\n--- Creating COMBINED dataset (50% SAIL + 50% Random) ---")
    comb_data = create_combined_dataset(sail_data, random_data, n_target, args.seed)
    comb_train, comb_val = split_train_val(comb_data, args.val_fraction, args.seed)
    datasets['combined'] = {'train': comb_train, 'val': comb_val}
    
    # Save datasets
    logger.info("\n--- Saving datasets ---")
    
    all_stats = {}
    for name, data in datasets.items():
        output_path = output_dir / f"dataset_{name}.npz"
        
        metadata = {
            'dataset_type': name,
            'n_samples': n_target,
            'val_fraction': args.val_fraction,
            'seed': args.seed,
        }
        
        save_dataset(output_path, data['train'], data['val'], metadata)
        
        # Compute and log statistics
        all_stats[name] = {
            'train': compute_dataset_statistics(data['train']),
            'val': compute_dataset_statistics(data['val'])
        }
    
    # Save statistics report
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)
    
    for name, stats in all_stats.items():
        train_stats = stats['train']
        val_stats = stats['val']
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Train: {train_stats['n_samples']} samples, "
                    f"obj={train_stats['objective_mean']:.2f}±{train_stats['objective_std']:.2f}")
        logger.info(f"  Val:   {val_stats['n_samples']} samples, "
                    f"obj={val_stats['objective_mean']:.2f}±{val_stats['objective_std']:.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
