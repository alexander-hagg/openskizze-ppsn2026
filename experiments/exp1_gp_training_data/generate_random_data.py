#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Phase 1B: Generate Random Training Data using Scrambled Sobol Sequences

This script generates random training data by:
1. Creating scrambled Sobol sequences in the 60-dimensional gene space
2. Evaluating each sample using KLAM_21 simulation
3. Saving genomes, objectives, and features for GP training

Usage:
    python experiments/generate_random_data.py --parcel-size 51 --num-samples 2000 --num-workers 128
    
For HPC SLURM array jobs:
    python experiments/generate_random_data.py --parcel-size $PARCEL_SIZE --num-samples 2000 --seed $SEED
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import multiprocessing as mp
from functools import partial
import time

import numpy as np
import yaml
from scipy.stats import qmc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from domain_description.evaluation_klam import eval, eval_multiple, calculate_planning_features
from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Parcel sizes from experiment plan (divisible by xy_scale=3)
PARCEL_SIZES = [60, 120, 240]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random training data using Sobol sequences"
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        required=True,
        choices=PARCEL_SIZES,
        help="Parcel size in meters (both width and height)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for KLAM_21 evaluation (default: CPU count)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for parallel evaluation (default: 128)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/random_data",
        help="Output directory for random data (default: results/random_data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--scramble",
        action="store_true",
        default=True,
        help="Use scrambled Sobol sequence (default: True)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip if output file already exists (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force regeneration even if output file exists (overrides --skip-existing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without running simulations (default: False)"
    )
    parser.add_argument(
        "--collect-spatial-data",
        action="store_true",
        default=False,
        help="Collect and save full spatial data (all timestamps, inputs, outputs) (default: False)"
    )
    return parser.parse_args()


def get_encoding_for_parcel_size(parcel_size_m: int) -> ParametricEncoding:
    """
    Create ParametricEncoding instance configured for the specified parcel size.
    
    Args:
        parcel_size_m: Parcel size in meters (both width and height)
        
    Returns:
        Configured ParametricEncoding instance
    """
    # Pass parcel_size directly to constructor - this sets both
    # self.length_design AND self.config['length_design'] correctly
    encoding = ParametricEncoding(parcel_size=parcel_size_m)
    
    logger.info(f"Configured encoding: length_design={encoding.length_design}, "
                f"parcel={parcel_size_m}x{parcel_size_m}m")
    
    return encoding


def generate_sobol_samples(
    num_samples: int,
    num_dimensions: int,
    seed: int,
    scramble: bool = True
) -> np.ndarray:
    """
    Generate samples using a (scrambled) Sobol sequence.
    
    The Sobol sequence is a quasi-random low-discrepancy sequence that provides
    better coverage of the sample space than purely random sampling.
    
    Args:
        num_samples: Number of samples to generate
        num_dimensions: Dimensionality of the gene space (60 for our encoding)
        seed: Random seed for scrambling
        scramble: Whether to use scrambled Sobol (recommended)
        
    Returns:
        np.ndarray of shape (num_samples, num_dimensions) with values in [0, 1]
    """
    logger.info(f"Generating {num_samples} Sobol samples in {num_dimensions}D space "
                f"(scrambled={scramble}, seed={seed})")
    
    # scipy.stats.qmc.Sobol requires power of 2 samples for optimal balance
    # We generate the next power of 2 and take the first num_samples
    n_power2 = 2 ** int(np.ceil(np.log2(num_samples)))
    
    # Use rng parameter for scipy >= 1.10
    rng = np.random.default_rng(seed) if scramble else None
    sampler = qmc.Sobol(d=num_dimensions, scramble=scramble, rng=rng)
    samples = sampler.random(n=n_power2)
    
    # Take first num_samples
    samples = samples[:num_samples]
    
    logger.info(f"Generated {len(samples)} samples, range: [{samples.min():.4f}, {samples.max():.4f}]")
    
    return samples


def evaluate_single_genome(
    genome: np.ndarray,
    config_environment: Dict[str, Any],
    config_encoding: Dict[str, Any],
    encoding: ParametricEncoding,
    sample_idx: int
) -> Tuple[int, float, np.ndarray]:
    """
    Evaluate a single genome using KLAM_21.
    
    Args:
        genome: 60-dimensional genome array
        config_environment: Environment configuration
        config_encoding: Encoding configuration
        encoding: Configured ParametricEncoding
        sample_idx: Index of this sample (for logging)
        
    Returns:
        Tuple of (sample_idx, fitness, features)
    """
    try:
        # Use the eval function from evaluation_klam
        result, _ = eval(
            solution=genome,
            config_environment=config_environment,
            config_encoding=config_encoding,
            solution_template=encoding,
            use_surrogate=False,
            debug=False
        )
        
        # Result format: [fitness, *features, *heightmap.flatten()]
        fitness = result[0]
        features = result[1:9]  # 8 features
        
        return (sample_idx, float(fitness), features)
        
    except Exception as e:
        logger.error(f"Sample {sample_idx} failed: {e}")
        return (sample_idx, np.nan, np.full(8, np.nan))


def evaluate_batch_parallel(
    genomes: np.ndarray,
    config_environment: Dict[str, Any],
    config_encoding: Dict[str, Any],
    encoding: ParametricEncoding,
    num_workers: int,
    batch_offset: int = 0,
    collect_spatial_data: bool = False
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Evaluate a batch of genomes in parallel using multiprocessing.
    
    Args:
        genomes: Array of shape (batch_size, 60)
        config_environment: Environment configuration
        config_encoding: Encoding configuration
        encoding: Configured ParametricEncoding
        num_workers: Number of parallel workers
        batch_offset: Starting index for this batch
        collect_spatial_data: Whether to collect full spatial data
        
    Returns:
        Tuple of (objectives, features, spatial_data_list)
    """
    batch_size = len(genomes)
    
    # For larger batches, use the built-in parallel evaluation
    if batch_size >= num_workers:
        try:
            # Create a pool for parallel evaluation
            with mp.Pool(processes=num_workers) as pool:
                results, _, spatial_data_list = eval_multiple(
                    solutions=genomes,
                    config_environment=config_environment,
                    config_encoding=config_encoding,
                    solution_template=encoding,
                    surrogate_model=None,
                    pool=pool,
                    debug=False,
                    collect_spatial_data=collect_spatial_data
                )
            
            # Results format: each row is [fitness, *features, *heightmap]
            objectives = results[:, 0]
            features = results[:, 1:9]  # 8 features
            
            return objectives, features, spatial_data_list
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Fall through to individual evaluation
    
    # Fallback: evaluate individually (for small batches or if batch fails)
    objectives = np.zeros(batch_size)
    features = np.zeros((batch_size, 8))
    spatial_data_list = [None] * batch_size if collect_spatial_data else None
    
    for i, genome in enumerate(genomes):
        idx, obj, feat = evaluate_single_genome(
            genome, config_environment, config_encoding, encoding, batch_offset + i
        )
        objectives[i] = obj
        features[i] = feat
        
    return objectives, features, spatial_data_list


def save_random_data(
    output_path: Path,
    genomes: np.ndarray,
    objectives: np.ndarray,
    features: np.ndarray,
    parcel_size_m: int,
    num_samples: int,
    seed: int,
    valid_mask: np.ndarray
):
    """
    Save generated random data to NPZ file.
    
    Args:
        output_path: Path to save NPZ file
        genomes: All generated genomes
        objectives: Objective values (cold air flux)
        features: Feature vectors (8-dimensional)
        parcel_size_m: Parcel size in meters
        num_samples: Number of samples requested
        seed: Random seed used
        valid_mask: Boolean mask of valid (non-NaN) samples
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter to valid samples only
    valid_genomes = genomes[valid_mask]
    valid_objectives = objectives[valid_mask]
    valid_features = features[valid_mask]
    
    # Widths and heights are just the parcel size (square parcels)
    n_valid = len(valid_genomes)
    widths = np.full(n_valid, parcel_size_m, dtype=np.float32)
    heights = np.full(n_valid, parcel_size_m, dtype=np.float32)
    
    np.savez(
        output_path,
        genomes=valid_genomes,
        widths=widths,
        heights=heights,
        objectives=valid_objectives,
        features=valid_features,
        parcel_size_m=parcel_size_m,
        num_samples_requested=num_samples,
        num_samples_valid=len(valid_genomes),
        seed=seed,
        source='sobol_scrambled'
    )
    
    logger.info(f"Saved {len(valid_genomes)} valid samples to {output_path}")
    logger.info(f"Objective range: [{valid_objectives.min():.4f}, {valid_objectives.max():.4f}]")
    logger.info(f"Mean objective: {valid_objectives.mean():.4f} ± {valid_objectives.std():.4f}")


def save_spatial_data(
    output_path: Path,
    spatial_data_list: List,
    parcel_size_m: int,
    num_samples: int,
    seed: int
):
    """
    Save spatial data for all samples to a compressed NPZ file.
    
    Spatial file structure:
    - Grid metadata (shared)
    - KLAM config (shared)
    - INPUTS per sample: terrain, buildings, landuse
    - OUTPUTS per sample per timestamp: uq, vq, uz, vz, Ex, Hx
    
    Args:
        output_path: Path to save the NPZ file
        spatial_data_list: List of spatial data dicts from evaluation
        parcel_size_m: Parcel size in meters
        num_samples: Number of samples requested
        seed: Random seed used
    """
    # Filter out None entries (failed evaluations)
    valid_samples = [(i, s) for i, s in enumerate(spatial_data_list) if s is not None]
    
    if not valid_samples:
        logger.warning("No valid spatial data to save!")
        return
    
    n_samples = len(valid_samples)
    first_sample = valid_samples[0][1]
    
    # Get grid shape from first sample
    grid_shape = first_sample['grid_shape']
    H, W = grid_shape
    timestamps = first_sample['timestamps']
    n_timestamps = len(timestamps)
    
    logger.info(f"Packing spatial data: {n_samples} samples, {grid_shape} grid, {n_timestamps} timestamps")
    
    # Initialize output arrays
    terrain = np.zeros((n_samples, H, W), dtype=np.float16)
    buildings = np.zeros((n_samples, H, W), dtype=np.float16)
    landuse = np.zeros((n_samples, H, W), dtype=np.int8)
    
    # Output fields: (N, T, H, W)
    uq_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    vq_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    uz_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    vz_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    ex_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    hx_all = np.zeros((n_samples, n_timestamps, H, W), dtype=np.float16)
    
    # Track original sample indices (for matching with genomes/objectives)
    sample_indices = np.zeros(n_samples, dtype=np.int32)
    
    # Pack data
    for i, (original_idx, s) in enumerate(valid_samples):
        sample_indices[i] = original_idx
        terrain[i] = s['terrain']
        buildings[i] = s['buildings']
        landuse[i] = s['landuse']
        
        if s['uq_all'] is not None:
            uq_all[i] = s['uq_all']
        if s['vq_all'] is not None:
            vq_all[i] = s['vq_all']
        if s['uz_all'] is not None:
            uz_all[i] = s['uz_all']
        if s['vz_all'] is not None:
            vz_all[i] = s['vz_all']
        if s['ex_all'] is not None:
            ex_all[i] = s['ex_all']
        if s['hx_all'] is not None:
            hx_all[i] = s['hx_all']
    
    # Save to compressed NPZ
    np.savez_compressed(
        output_path,
        # Metadata
        n_samples=n_samples,
        grid_shape=np.array(grid_shape),
        parcel_offset=np.array(first_sample['parcel_offset']),
        parcel_size_cells=np.array(first_sample['parcel_size']),
        cell_size=first_sample['cell_size'],
        timestamps=np.array(timestamps),
        
        # KLAM config
        wind_speed=first_sample['wind_speed'],
        wind_direction=first_sample['wind_direction'],
        sim_duration=first_sample['sim_duration'],
        
        # Sample index mapping
        sample_indices=sample_indices,
        
        # INPUTS (N, H, W)
        terrain=terrain,
        buildings=buildings,
        landuse=landuse,
        
        # OUTPUTS (N, T, H, W) - raw KLAM units (cm/s, 1/10 m, 100 J/m²)
        uq=uq_all,
        vq=vq_all,
        uz=uz_all,
        vz=vz_all,
        Ex=ex_all,
        Hx=hx_all,
        
        # Additional metadata
        parcel_size_m=parcel_size_m,
        num_samples_requested=num_samples,
        seed=seed,
        source='sobol_scrambled'
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved spatial data to: {output_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # Calculate compression ratio for information
    uncompressed_mb = (
        terrain.nbytes + buildings.nbytes + landuse.nbytes +
        uq_all.nbytes + vq_all.nbytes + uz_all.nbytes + 
        vz_all.nbytes + ex_all.nbytes + hx_all.nbytes
    ) / (1024 * 1024)
    logger.info(f"Compression ratio: {file_size_mb/uncompressed_mb:.1%} of {uncompressed_mb:.1f} MB uncompressed")


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Phase 1B: Random Data Generation (Sobol Sequence)")
    logger.info("=" * 60)
    logger.info(f"Parcel size: {args.parcel_size}x{args.parcel_size} m")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Scrambled Sobol: {args.scramble}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Skip existing: {args.skip_existing and not args.force}")
    logger.info(f"Force regenerate: {args.force}")
    logger.info(f"Collect spatial data: {args.collect_spatial_data}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Determine output file path
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"random_sobol_{args.parcel_size}m_n{args.num_samples}_seed{args.seed}.npz"
    spatial_file = output_dir / f"random_sobol_{args.parcel_size}m_n{args.num_samples}_seed{args.seed}_spatial.npz"
    
    # Check if output already exists (applies to both normal and dry-run mode)
    file_exists = output_file.exists()
    existing_valid = False
    existing_samples = 0
    
    if file_exists:
        try:
            existing = np.load(output_file)
            existing_samples = len(existing['objectives'])
            existing_valid = True
        except Exception as e:
            logger.warning(f"Existing file is corrupted: {e}")
            existing_valid = False
    
    # Dry run mode - test everything including skip logic
    if args.dry_run:
        logger.info("")
        logger.info("=" * 60)
        logger.info("DRY RUN - No simulations will be executed")
        logger.info("=" * 60)
        logger.info(f"Would generate: {args.num_samples} samples")
        logger.info(f"Output file: {output_file}")
        logger.info(f"File exists: {file_exists}")
        
        if file_exists and existing_valid:
            logger.info(f"Existing file samples: {existing_samples}")
            if args.force:
                logger.info("→ WOULD OVERWRITE: --force is enabled")
            elif args.skip_existing:
                logger.info("→ WOULD SKIP: --skip-existing is enabled (default) and file is valid")
            else:
                logger.info("→ WOULD OVERWRITE: --skip-existing is disabled")
        elif file_exists and not existing_valid:
            logger.info("Existing file is CORRUPTED")
            logger.info("→ WOULD REGENERATE: File exists but is not valid")
        else:
            logger.info("→ WOULD GENERATE: No existing file found")
        
        logger.info(f"Workers: {args.num_workers or mp.cpu_count()}")
        logger.info(f"Batch size: {args.batch_size}")
        
        # Test import and encoding
        try:
            encoding = get_encoding_for_parcel_size(args.parcel_size)
            num_genes = encoding.get_dimension()
            logger.info(f"Encoding dimension: {num_genes}")
            logger.info("✓ Imports and encoding validated successfully")
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            return 1
        
        logger.info("")
        logger.info("DRY RUN COMPLETE - Use without --dry-run to execute")
        return 0
    
    # Normal mode: apply skip logic (skip unless --force is used)
    if args.skip_existing and not args.force and file_exists and existing_valid:
        logger.info(f"SKIPPING: Output file already exists with {existing_samples} samples")
        logger.info(f"  File: {output_file}")
        logger.info(f"  Use --force to regenerate")
        return 0  # Success - nothing to do
    
    if file_exists and not existing_valid:
        logger.warning(f"Existing file is corrupted, will regenerate")
    
    # Setup encoding
    encoding = get_encoding_for_parcel_size(args.parcel_size)
    num_genes = encoding.get_dimension()  # Should be 60
    
    # Load configs
    with open(project_root / "domain_description/cfg.yml") as f:
        config_environment = yaml.safe_load(f)
    
    config_encoding = encoding.config.copy()
    
    # Force KLAM evaluation
    config_environment['evaluation_method'] = 'klam'
    
    # Setup workers
    num_workers = args.num_workers or mp.cpu_count()
    logger.info(f"Using {num_workers} workers for parallel evaluation")
    
    # Generate Sobol samples
    genomes = generate_sobol_samples(
        num_samples=args.num_samples,
        num_dimensions=num_genes,
        seed=args.seed,
        scramble=args.scramble
    )
    
    # Scale genomes from [0, 1] to [-1, 1] (the expected gene range)
    genomes = 2 * genomes - 1
    
    # Initialize result arrays
    all_objectives = np.zeros(args.num_samples)
    all_features = np.zeros((args.num_samples, 8))
    all_spatial_data = [None] * args.num_samples if args.collect_spatial_data else None
    
    # Evaluate in batches
    start_time = time.time()
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_genomes = genomes[batch_start:batch_end]
        
        logger.info(f"Evaluating batch {batch_idx + 1}/{num_batches} "
                    f"(samples {batch_start}-{batch_end-1})")
        
        batch_objectives, batch_features, batch_spatial = evaluate_batch_parallel(
            genomes=batch_genomes,
            config_environment=config_environment,
            config_encoding=config_encoding,
            encoding=encoding,
            num_workers=num_workers,
            batch_offset=batch_start,
            collect_spatial_data=args.collect_spatial_data
        )
        
        all_objectives[batch_start:batch_end] = batch_objectives
        all_features[batch_start:batch_end] = batch_features
        
        # Store spatial data if collected
        if args.collect_spatial_data and batch_spatial is not None:
            for i, spatial_data in enumerate(batch_spatial):
                all_spatial_data[batch_start + i] = spatial_data
        
        # Progress update
        elapsed = time.time() - start_time
        samples_done = batch_end
        rate = samples_done / elapsed if elapsed > 0 else 0
        eta = (args.num_samples - samples_done) / rate if rate > 0 else 0
        
        logger.info(f"Progress: {samples_done}/{args.num_samples} samples "
                    f"({rate:.2f} samples/s, ETA: {eta/60:.1f} min)")
    
    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time/60:.2f} minutes")
    
    # Check for failed evaluations
    valid_mask = ~np.isnan(all_objectives)
    num_valid = valid_mask.sum()
    num_failed = args.num_samples - num_valid
    
    if num_failed > 0:
        logger.warning(f"{num_failed} samples failed evaluation ({num_failed/args.num_samples*100:.1f}%)")
    
    logger.info(f"Valid samples: {num_valid}/{args.num_samples}")
    
    # Save results (output_file already defined at the top of main)
    save_random_data(
        output_path=output_file,
        genomes=genomes,
        objectives=all_objectives,
        features=all_features,
        parcel_size_m=args.parcel_size,
        num_samples=args.num_samples,
        seed=args.seed,
        valid_mask=valid_mask
    )
    
    # Save spatial data if collected
    if args.collect_spatial_data and all_spatial_data is not None:
        logger.info("Saving spatial data...")
        save_spatial_data(
            output_path=spatial_file,
            spatial_data_list=all_spatial_data,
            parcel_size_m=args.parcel_size,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    logger.info("=" * 60)
    logger.info("Random data generation complete!")
    if args.collect_spatial_data:
        logger.info(f"Spatial data saved to: {spatial_file}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
