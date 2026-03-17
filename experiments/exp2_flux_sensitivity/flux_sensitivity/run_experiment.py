#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Flux Sensitivity Experiment: Run KLAM_21 Evaluations

Full factorial experiment testing cold air flux response to:
- Site Coverage (GRZ): 0%, 20%, 40%, 60%, 80%, 100%
- Building Height: 0, 2, 4, 6, 8 floors
- Orientation: 0°, 45°, 90° relative to wind direction

Uses deterministic "bar pattern" layouts for comparability across configurations.

Usage:
    python experiments/flux_sensitivity/run_experiment.py
    python experiments/flux_sensitivity/run_experiment.py --num-workers 32
    python experiments/flux_sensitivity/run_experiment.py --dry-run
"""

import argparse
import os
import sys
import yaml
import numpy as np
import multiprocessing
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import product
from tqdm import tqdm
from scipy.ndimage import rotate

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from domain_description.evaluation_klam import (
    compute_fitness_klam, calculate_planning_features, init_environment
)


# =============================================================================
# EXPERIMENTAL DESIGN
# =============================================================================

# Factor levels
GRZ_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Site coverage ratios
HEIGHT_LEVELS = [0, 2, 4, 6, 8]  # Floors
ORIENTATION_LEVELS = [0, 45, 90]  # Degrees relative to wind

# Fixed parameters
PARCEL_CELLS = 17  # 51m at 3m/cell
XY_SCALE = 3.0  # meters per cell
Z_SCALE = 3.0  # meters per floor


# =============================================================================
# DETERMINISTIC LAYOUT GENERATION
# =============================================================================

def create_bar_layout(grz: float, height_floors: int, orientation_deg: int, 
                      parcel_cells: int = PARCEL_CELLS) -> np.ndarray:
    """
    Create a deterministic bar pattern layout.
    
    The pattern consists of parallel rectangular bars that:
    - Achieve the target GRZ (site coverage)
    - Are all at the same height
    - Are oriented at the specified angle relative to the grid (0=E-W, 90=N-S)
    
    Args:
        grz: Target site coverage (0.0 to 1.0)
        height_floors: Building height in floors (0-8)
        orientation_deg: Bar orientation in degrees (0, 45, or 90)
        parcel_cells: Grid size in cells
        
    Returns:
        heightmap: (parcel_cells, parcel_cells) array of floor counts
    """
    heightmap = np.zeros((parcel_cells, parcel_cells), dtype=np.float32)
    
    # Handle edge cases
    if grz <= 0.0 or height_floors <= 0:
        return heightmap
    
    total_cells = parcel_cells * parcel_cells
    target_occupied = int(grz * total_cells)
    
    if target_occupied <= 0:
        return heightmap
    
    # For 100% coverage, just fill everything
    if grz >= 1.0:
        heightmap[:, :] = height_floors
        return heightmap
    
    # For 80% coverage, use a block with a corridor through the middle
    if grz >= 0.75:
        # Create a solid block with a gap/corridor
        # Gap width scales inversely with GRZ
        gap_width = max(1, int((1.0 - grz) * parcel_cells))
        gap_start = (parcel_cells - gap_width) // 2
        
        heightmap[:, :] = height_floors
        # Cut a horizontal corridor through the middle
        if orientation_deg == 0:
            # Horizontal corridor (parallel to wind) - most permeable
            heightmap[gap_start:gap_start+gap_width, :] = 0
        elif orientation_deg == 90:
            # Vertical corridor (perpendicular to wind) - least permeable
            heightmap[:, gap_start:gap_start+gap_width] = 0
        else:  # 45 degrees
            # Diagonal corridor
            for i in range(parcel_cells):
                j_center = i  # Diagonal line
                j_start = max(0, j_center - gap_width // 2)
                j_end = min(parcel_cells, j_center + gap_width // 2 + 1)
                heightmap[i, j_start:j_end] = 0
        return heightmap
    
    # For lower densities, use bar pattern
    # Bar width is fixed at 3 cells
    bar_width = 3
    
    # Calculate number of bars based on target GRZ
    if grz <= 0.25:
        num_bars = 2
    elif grz <= 0.5:
        num_bars = 3
    else:  # 0.5 < grz <= 0.75
        num_bars = 4
    
    # Calculate bar length to achieve target GRZ
    # GRZ = (num_bars * bar_width * bar_length) / total_cells
    bar_length = int((grz * total_cells) / (num_bars * bar_width))
    bar_length = min(bar_length, parcel_cells)  # Can't exceed parcel size
    bar_length = max(bar_length, 3)  # Minimum 3 cells
    
    # Calculate spacing between bars (ensure they fit)
    total_bar_height = num_bars * bar_width
    if total_bar_height >= parcel_cells:
        # Too many bars, reduce number
        num_bars = (parcel_cells - 2) // (bar_width + 1)
        total_bar_height = num_bars * bar_width
    
    available_space = parcel_cells - total_bar_height
    spacing = available_space // (num_bars + 1)
    spacing = max(1, spacing)
    
    # Create the base pattern (horizontal bars, 0° orientation)
    base_pattern = np.zeros((parcel_cells, parcel_cells), dtype=np.float32)
    
    current_y = spacing
    for i in range(num_bars):
        if current_y + bar_width > parcel_cells:
            break
        # Center the bar horizontally
        x_start = (parcel_cells - bar_length) // 2
        x_end = x_start + bar_length
        y_end = min(current_y + bar_width, parcel_cells)
        base_pattern[current_y:y_end, x_start:x_end] = 1.0
        current_y += bar_width + spacing
    
    # Apply rotation if needed
    if orientation_deg == 0:
        rotated = base_pattern
    elif orientation_deg == 45:
        # Rotate 45 degrees
        # Use scipy.ndimage.rotate with reshape=False to keep same dimensions
        rotated = rotate(base_pattern, 45, reshape=False, order=0, mode='constant', cval=0)
        rotated = (rotated > 0.5).astype(np.float32)
    elif orientation_deg == 90:
        # Rotate 90 degrees (transpose)
        rotated = np.rot90(base_pattern, k=1)
    else:
        rotated = base_pattern
    
    # Apply height
    heightmap = rotated * height_floors
    
    return heightmap


def generate_all_configurations() -> List[Dict]:
    """
    Generate all experimental configurations.
    
    Returns:
        List of configuration dictionaries with keys:
        - grz, height_floors, orientation_deg
        - config_id (unique identifier string)
    """
    configurations = []
    config_id = 0
    
    for grz, height, orientation in product(GRZ_LEVELS, HEIGHT_LEVELS, ORIENTATION_LEVELS):
        # Skip degenerate cases (0 height = empty regardless of GRZ/orientation)
        if height == 0 and (grz > 0 or orientation > 0):
            # Only keep one empty case
            if not (grz == 0 and orientation == 0):
                continue
        
        # Skip 0% GRZ with different orientations (all identical)
        if grz == 0 and orientation > 0:
            continue
            
        config = {
            'config_id': config_id,
            'grz': grz,
            'height_floors': height,
            'orientation_deg': orientation,
            'label': f"GRZ{int(grz*100):03d}_H{height}_O{orientation:02d}"
        }
        configurations.append(config)
        config_id += 1
    
    return configurations


# =============================================================================
# EVALUATION WRAPPER
# =============================================================================

def evaluate_configuration(config: Dict, config_environment: Dict, 
                          config_encoding: Dict, collect_spatial: bool = True) -> Dict:
    """
    Evaluate a single configuration with KLAM_21.
    
    Args:
        config: Configuration dict with grz, height_floors, orientation_deg
        config_environment: Environment configuration
        config_encoding: Encoding configuration
        collect_spatial: Whether to collect full spatial data
        
    Returns:
        Result dict with fitness, features, heightmap, and optional spatial data
    """
    # Generate the heightmap
    heightmap_floors = create_bar_layout(
        grz=config['grz'],
        height_floors=config['height_floors'],
        orientation_deg=config['orientation_deg'],
        parcel_cells=config_encoding['length_design']
    )
    
    # Convert to meters for feature calculation
    heightmap_meters = heightmap_floors * Z_SCALE
    
    # Calculate planning features
    features = calculate_planning_features(heightmap_meters, config_encoding)
    
    # Create voxel representation for KLAM
    max_floors = config_encoding.get('max_num_floors', 10)
    length = config_encoding['length_design']
    voxels = np.zeros((length, length, max_floors), dtype=np.int32)
    for x in range(length):
        for y in range(length):
            for z in range(int(heightmap_floors[x, y])):
                voxels[y, x, z] = 1
    
    # Run KLAM_21 simulation
    try:
        sim_results = compute_fitness_klam(
            config_environment, 
            config_encoding, 
            voxels, 
            heightmap_floors,
            debug=False,
            collect_spatial_data=collect_spatial
        )
        fitness = sim_results['fitness']
        spatial_data = sim_results.get('spatial_data') if collect_spatial else None
        uq = sim_results.get('uq')
        vq = sim_results.get('vq')
        success = True
    except Exception as e:
        print(f"  WARNING: Config {config['label']} failed: {e}")
        fitness = np.nan
        spatial_data = None
        uq, vq = None, None
        success = False
    
    return {
        'config': config,
        'heightmap_floors': heightmap_floors,
        'heightmap_meters': heightmap_meters,
        'features': features,
        'fitness': fitness,
        'uq': uq,
        'vq': vq,
        'spatial_data': spatial_data,
        'success': success
    }


def _eval_config_wrapper(args):
    """Wrapper for multiprocessing."""
    idx, config, config_environment, config_encoding, collect_spatial = args
    result = evaluate_configuration(config, config_environment, config_encoding, collect_spatial)
    return idx, result


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment(output_dir: str, num_workers: int = None, 
                   collect_spatial: bool = True, dry_run: bool = False):
    """
    Run the full factorial experiment.
    
    Args:
        output_dir: Directory to save results
        num_workers: Number of parallel workers (default: all CPUs)
        collect_spatial: Whether to collect full spatial flux data
        dry_run: If True, just print configuration summary without running
    """
    print("=" * 70)
    print("FLUX SENSITIVITY EXPERIMENT")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate configurations
    configurations = generate_all_configurations()
    print(f"Experimental Design:")
    print(f"  GRZ levels: {GRZ_LEVELS}")
    print(f"  Height levels: {HEIGHT_LEVELS} floors")
    print(f"  Orientation levels: {ORIENTATION_LEVELS}°")
    print(f"  Total configurations: {len(configurations)}")
    print()
    
    if dry_run:
        print("DRY RUN - Configuration summary:")
        print("-" * 70)
        for config in configurations:
            print(f"  [{config['config_id']:3d}] {config['label']}: "
                  f"GRZ={config['grz']:.0%}, H={config['height_floors']}fl, "
                  f"O={config['orientation_deg']}°")
        print("-" * 70)
        print(f"Would evaluate {len(configurations)} configurations.")
        return
    
    # Load configurations
    with open(project_root / "domain_description/cfg.yml") as f:
        config_environment = yaml.safe_load(f)
    with open(project_root / "encodings/parametric/cfg.yml") as f:
        config_encoding = yaml.safe_load(f)
    
    # Initialize environment
    config_environment = init_environment(config_environment)
    
    # Set parcel size (51m = 17 cells at 3m/cell)
    config_encoding['length_design'] = PARCEL_CELLS
    config_encoding['xy_scale'] = XY_SCALE
    config_encoding['z_scale'] = Z_SCALE
    
    print(f"Parcel: {PARCEL_CELLS}×{PARCEL_CELLS} cells = "
          f"{PARCEL_CELLS * XY_SCALE:.0f}m × {PARCEL_CELLS * XY_SCALE:.0f}m")
    print(f"Collect spatial data: {collect_spatial}")
    print()
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup workers
    if num_workers is None:
        num_workers = min(psutil.cpu_count(logical=True), len(configurations))
    print(f"Using {num_workers} workers")
    print()
    
    # Run evaluations
    print("Running KLAM_21 evaluations...")
    print("-" * 70)
    
    results = [None] * len(configurations)
    
    if num_workers > 1:
        # Parallel evaluation
        args_list = [
            (i, config, config_environment, config_encoding, collect_spatial)
            for i, config in enumerate(configurations)
        ]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            for idx, result in tqdm(pool.imap_unordered(_eval_config_wrapper, args_list),
                                   total=len(configurations), desc="Evaluating"):
                results[idx] = result
    else:
        # Sequential evaluation
        for i, config in enumerate(tqdm(configurations, desc="Evaluating")):
            results[i] = evaluate_configuration(
                config, config_environment, config_encoding, collect_spatial
            )
    
    print("-" * 70)
    
    # Count successes
    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success
    print(f"Completed: {n_success} successful, {n_failed} failed")
    print()
    
    # Prepare data for saving
    n_configs = len(configurations)
    n_features = 8
    
    # Arrays for main data
    config_ids = np.array([r['config']['config_id'] for r in results])
    grz_values = np.array([r['config']['grz'] for r in results])
    height_values = np.array([r['config']['height_floors'] for r in results])
    orientation_values = np.array([r['config']['orientation_deg'] for r in results])
    labels = np.array([r['config']['label'] for r in results])
    
    fitness_values = np.array([r['fitness'] for r in results])
    features_array = np.stack([r['features'] for r in results])
    heightmaps = np.stack([r['heightmap_floors'] for r in results])
    
    # Save main results
    output_path = Path(output_dir) / "flux_sensitivity_results.npz"
    
    save_dict = {
        'config_ids': config_ids,
        'grz': grz_values,
        'height_floors': height_values,
        'orientation_deg': orientation_values,
        'labels': labels,
        'fitness': fitness_values,
        'features': features_array,
        'heightmaps': heightmaps,
        'feature_names': np.array([
            'GRZ', 'GFZ', 'Avg Height (m)', 'Height Variability (m)',
            'Avg Distance (m)', 'Building Count', 'Compactness', 'Park Factor (m)'
        ]),
        'parcel_cells': PARCEL_CELLS,
        'xy_scale': XY_SCALE,
        'z_scale': Z_SCALE,
    }
    
    np.savez(output_path, **save_dict)
    print(f"Saved main results to: {output_path}")
    
    # Save spatial data separately (can be large)
    if collect_spatial:
        spatial_path = Path(output_dir) / "flux_sensitivity_spatial.npz"
        
        # Collect non-None spatial data
        uq_list = []
        vq_list = []
        spatial_indices = []
        
        for i, r in enumerate(results):
            if r['uq'] is not None and r['vq'] is not None:
                uq_list.append(r['uq'])
                vq_list.append(r['vq'])
                spatial_indices.append(i)
        
        if uq_list:
            np.savez(
                spatial_path,
                uq=np.stack(uq_list),
                vq=np.stack(vq_list),
                config_indices=np.array(spatial_indices),
                labels=labels[spatial_indices]
            )
            print(f"Saved spatial data to: {spatial_path}")
    
    # Print summary statistics
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    valid_fitness = fitness_values[~np.isnan(fitness_values)]
    if len(valid_fitness) > 0:
        print(f"Fitness (Cold Air Flux):")
        print(f"  Min:  {np.min(valid_fitness):.4f}")
        print(f"  Max:  {np.max(valid_fitness):.4f}")
        print(f"  Mean: {np.mean(valid_fitness):.4f}")
        print(f"  Std:  {np.std(valid_fitness):.4f}")
        print()
        
        # Find extremes
        min_idx = np.nanargmin(fitness_values)
        max_idx = np.nanargmax(fitness_values)
        print(f"Lowest flux:  {labels[min_idx]} = {fitness_values[min_idx]:.4f}")
        print(f"Highest flux: {labels[max_idx]} = {fitness_values[max_idx]:.4f}")
    
    print()
    print(f"Results saved to: {output_dir}")
    print("Run analysis script for detailed visualization and statistics.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run flux sensitivity factorial experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flux_sensitivity",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPUs)"
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip collecting spatial flux data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration summary without running"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        collect_spatial=not args.no_spatial,
        dry_run=args.dry_run
    )
