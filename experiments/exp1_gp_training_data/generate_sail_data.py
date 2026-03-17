#!/usr/bin/env python
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Phase 1A: Generate SAIL optimized training data for GP surrogate.

STAGED PIPELINE:
1. Check if final _klam.npz output exists → SKIP ALL
2. Check if old surrogate-based .npz exists → Load genomes (skip SAIL)
3. Check if FinalQD_archive.pkl exists → Load genomes (skip SAIL)
4. If none exist → Run SAIL
5. Evaluate ALL elites with REAL KLAM_21 (not surrogate!)
6. Save to sail_{size}x{size}_rep{N}_klam.npz

IMPORTANT: Old .npz files and SAIL archives contain SURROGATE-predicted fitness!
This script re-evaluates all elites with real KLAM_21 for accurate GP training data.

Usage:
    # Full run (HPC)
    python experiments/generate_sail_data.py --parcel-size 51 --num-generations 100000 \
        --num-workers 128 --output-dir results/gp_experiment/sail_data --seed 42
    
    # Quick test
    python experiments/generate_sail_data.py --parcel-size 51 --num-generations 100 \
        --num-workers 4 --output-dir results/gp_experiment/sail_data_test --seed 42
    
    # Dry run to check skip logic
    python experiments/generate_sail_data.py --parcel-size 51 --output-dir results/sail_data --dry-run
"""

import argparse
import sys
import os
import numpy as np
import yaml
import pickle
from pathlib import Path
from datetime import datetime
import multiprocessing
import psutil
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from optimization.sail_optimizer import run_sail_optimization
from domain_description.evaluation_klam import eval_multiple


def extract_archive_data(archive, parcel_size: float):
    """
    Extract all elite solutions from a PyRibs archive.
    
    NOTE: The 'objectives' from the archive are SURROGATE predictions, not real KLAM_21!
    We extract them for reference but will recalculate with real KLAM_21.
    
    Returns:
        dict with keys: genomes, widths, heights, surrogate_objectives, features, phenotypes
    """
    # Get all data from archive
    data = archive.data()
    
    n_elites = len(data['objective'])
    
    if n_elites == 0:
        return None
    
    # Extract fields
    genomes = np.array(data['solution'])  # (N, 60)
    surrogate_objectives = np.array(data['objective'])  # (N,) - SURROGATE predictions!
    features = np.array(data['measures'])  # (N, 8)
    phenotypes = np.array(data['heightmaps'])  # (N, L*L)
    
    # Create size arrays
    widths = np.full(n_elites, parcel_size, dtype=np.float32)
    heights = np.full(n_elites, parcel_size, dtype=np.float32)
    
    return {
        'genomes': genomes.astype(np.float32),
        'widths': widths,
        'heights': heights,
        'surrogate_objectives': surrogate_objectives.astype(np.float32),  # Renamed for clarity
        'features': features.astype(np.float32),
        'phenotypes': phenotypes.astype(np.float32)
    }


def load_archive_from_pickle(archive_path: Path):
    """Load a PyRibs archive from pickle file."""
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    return archive


def check_sail_completion(run_dir: Path, target_generations: int):
    """
    Check if SAIL optimization completed to target generation count.
    
    SAIL has TWO phases:
    1. SAIL phase: 0 to num_generations (with GP training)
    2. FinalQD phase: 0 to num_generations (pure exploitation)
    
    Total expected generations = 2 × num_generations = 2 × target_generations
    
    Args:
        run_dir: Directory containing SAIL output files
        target_generations: Target number of generations PER PHASE
    
    Returns:
        tuple: (completed: bool, total_gen: int, sail_complete: bool, finalqd_gen: int)
            - completed: True if FinalQD reached target_generations
            - total_gen: SAIL_gens + FinalQD_gens (for resume_generation parameter)
            - sail_complete: True if SAIL phase is done
            - finalqd_gen: Current generation within FinalQD phase
    """
    # Check SAIL phase completion first
    sail_stats_files = list(run_dir.glob('*SAIL_stats.pkl'))
    sail_gen = 0
    sail_complete = False
    
    # Check FinalQD phase - if FinalQD files exist, SAIL must be complete
    finalqd_stats_files = list(run_dir.glob('*FinalQD_stats.pkl'))
    finalqd_archive_files = list(run_dir.glob('*FinalQD_archive.pkl'))
    finalqd_gen = 0
    finalqd_complete = False
    
    # If FinalQD archive exists, SAIL phase is definitely complete
    if finalqd_archive_files:
        sail_complete = True
        sail_gen = target_generations
        print(f"  Debug: FinalQD archive exists → SAIL phase complete")
    elif sail_stats_files:
        sail_stats_path = sail_stats_files[0]
        try:
            with open(sail_stats_path, 'rb') as f:
                sail_stats = pickle.load(f)
            
            # Stats is a list of ArchiveStats (one entry per generation)
            if isinstance(sail_stats, list) and len(sail_stats) > 0:
                sail_gen = len(sail_stats) - 1
                print(f"  Debug: SAIL_stats has {len(sail_stats)} entries → generation {sail_gen}")
            
            sail_complete = sail_gen >= target_generations
        except Exception as e:
            print(f"  Warning: Could not read SAIL stats file: {e}")
    
    # Check FinalQD phase progress
    if finalqd_stats_files:
        finalqd_stats_path = finalqd_stats_files[0]
        try:
            with open(finalqd_stats_path, 'rb') as f:
                finalqd_stats = pickle.load(f)
            
            # Stats is a list of ArchiveStats (one entry per generation)
            if isinstance(finalqd_stats, list) and len(finalqd_stats) > 0:
                finalqd_gen = len(finalqd_stats) - 1
                print(f"  Debug: FinalQD_stats has {len(finalqd_stats)} entries → generation {finalqd_gen}")
            
            finalqd_complete = finalqd_gen >= target_generations
        except Exception as e:
            print(f"  Warning: Could not read FinalQD stats file: {e}")
    
    # Calculate total generation for resume_generation parameter
    # sail_optimizer expects resume_generation = SAIL_gens + FinalQD_gens
    if sail_complete:
        total_gen = target_generations + finalqd_gen
    else:
        total_gen = sail_gen
    
    # Overall completion means FinalQD finished
    completed = sail_complete and finalqd_complete
    
    return completed, total_gen, sail_complete, finalqd_gen


def load_data_from_npz(npz_path: Path):
    """
    Load elite data from an existing .npz file (surrogate-evaluated).
    
    Returns:
        dict with keys: genomes, widths, heights, surrogate_objectives, features, phenotypes
        or None if file is invalid
    """
    try:
        data = np.load(npz_path)
        return {
            'genomes': data['genomes'],
            'widths': data['widths'],
            'heights': data['heights'],
            'surrogate_objectives': data['objectives'],  # These are surrogate predictions!
            'features': data['features'],
            'phenotypes': data['phenotypes'],
            'parcel_size': int(data['parcel_size']),
            'replicate': int(data['replicate']),
            'num_generations': int(data['num_generations']),
            'seed': int(data['seed'])
        }
    except Exception as e:
        print(f"  ⚠ Failed to load {npz_path}: {e}")
        return None


def evaluate_elites_with_klam(genomes, config_environment, config_encoding, solution_template, 
                              num_workers=None, batch_size=100, collect_spatial_data=False):
    """
    Evaluate all elite genomes with real KLAM_21 simulation.
    
    Args:
        genomes: (N, D) array of elite genomes
        config_environment: Environment configuration
        config_encoding: Encoding configuration
        solution_template: Solution template for phenotype expression
        num_workers: Number of parallel workers (default: all CPUs)
        batch_size: Batch size for parallel evaluation
        collect_spatial_data: If True, also return full spatial data for all samples
    
    Returns:
        objectives: (N,) array of real KLAM_21 fitness values (NaN for failed samples)
        spatial_data_list: List of spatial data dicts (only if collect_spatial_data=True)
    """
    n_elites = len(genomes)
    
    # Set up parallel pool
    if num_workers is None:
        num_workers = psutil.cpu_count(logical=True)
    
    print(f"  Evaluating {n_elites:,} elites with real KLAM_21 using {num_workers} workers...")
    if collect_spatial_data:
        print("  Collecting full spatial data (all timestamps, inputs, and outputs)")
    
    objectives = np.full(n_elites, np.nan, dtype=np.float32)
    all_spatial_data = [None] * n_elites if collect_spatial_data else None
    
    # Process in batches with progress bar
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_start in tqdm(range(0, n_elites, batch_size), desc="  KLAM evaluation"):
            batch_end = min(batch_start + batch_size, n_elites)
            batch_genomes = genomes[batch_start:batch_end]
            
            try:
                # surrogate_model=None → uses real KLAM_21
                results, _, spatial_data_list = eval_multiple(
                    batch_genomes,
                    config_environment,
                    config_encoding,
                    solution_template,
                    surrogate_model=None,  # CRITICAL: Real KLAM_21, not surrogate!
                    pool=pool,
                    debug=False,
                    collect_spatial_data=collect_spatial_data
                )
                
                # Extract fitness (column 0)
                batch_objectives = results[:, 0]
                objectives[batch_start:batch_end] = batch_objectives
                
                # Store spatial data if collected
                if collect_spatial_data and spatial_data_list is not None:
                    for i, spatial_data in enumerate(spatial_data_list):
                        all_spatial_data[batch_start + i] = spatial_data
                
            except Exception as e:
                print(f"    WARNING: Batch {batch_start}-{batch_end} failed: {e}")
                # Leave as NaN for this batch
    
    # Count successful evaluations
    n_valid = np.sum(~np.isnan(objectives))
    n_failed = np.sum(np.isnan(objectives))
    
    print(f"  Evaluation complete: {n_valid:,} successful, {n_failed:,} failed (NaN)")
    
    if collect_spatial_data:
        n_spatial = sum(1 for s in all_spatial_data if s is not None)
        print(f"  Spatial data collected for {n_spatial:,} samples")
        return objectives, all_spatial_data
    else:
        return objectives, None


def save_spatial_data(output_path, spatial_data_list, parcel_size, replicate):
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
        parcel_size: Parcel size in meters
        replicate: Replicate number
    """
    # Filter out None entries (failed evaluations)
    valid_samples = [(i, s) for i, s in enumerate(spatial_data_list) if s is not None]
    
    if not valid_samples:
        print("  WARNING: No valid spatial data to save!")
        return
    
    n_samples = len(valid_samples)
    first_sample = valid_samples[0][1]
    
    # Get grid shape from first sample
    grid_shape = first_sample['grid_shape']
    H, W = grid_shape
    timestamps = first_sample['timestamps']
    n_timestamps = len(timestamps)
    
    print(f"  Packing spatial data: {n_samples} samples, {grid_shape} grid, {n_timestamps} timestamps")
    
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
        parcel_size_m=parcel_size,
        replicate=replicate,
    )
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved spatial data to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Calculate compression ratio for information
    uncompressed_mb = (
        terrain.nbytes + buildings.nbytes + landuse.nbytes +
        uq_all.nbytes + vq_all.nbytes + uz_all.nbytes + 
        vz_all.nbytes + ex_all.nbytes + hx_all.nbytes
    ) / (1024 * 1024)
    print(f"  Compression ratio: {file_size_mb/uncompressed_mb:.1%} of {uncompressed_mb:.1f} MB uncompressed")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SAIL optimized training data for GP surrogate (with real KLAM_21 evaluation)'
    )
    parser.add_argument('--parcel-size', type=int, required=True,
                        help='Parcel size in meters (must be divisible by 3)')
    parser.add_argument('--replicate', type=int, default=1,
                        help='Replicate number (for multiple runs per size)')
    parser.add_argument('--num-generations', type=int, default=100000,
                        help='Number of SAIL generations')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output NPZ file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip if output file already exists (default: True)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force regeneration even if output file exists')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Show what would be done without running SAIL')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for KLAM evaluation (default: 100)')
    parser.add_argument('--collect-spatial-data', action='store_true', default=False,
                        help='Collect and save full spatial data (all timestamps, inputs, outputs)')
    
    args = parser.parse_args()
    
    # Validate parcel size
    if args.parcel_size % 3 != 0:
        print(f"ERROR: Parcel size {args.parcel_size} not divisible by xy_scale=3")
        sys.exit(1)
    
    design_cells = args.parcel_size // 3
    
    print("="*70)
    print("SAIL DATA GENERATION (with real KLAM_21 evaluation)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Parcel size: {args.parcel_size}×{args.parcel_size} m")
    print(f"  Design cells: {design_cells}×{design_cells}")
    print(f"  Replicate: {args.replicate}")
    print(f"  Generations: {args.num_generations:,}")
    print(f"  Workers: {args.num_workers or 'all CPUs'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Skip existing: {args.skip_existing and not args.force}")
    print(f"  Force regenerate: {args.force}")
    print(f"  Collect spatial data: {args.collect_spatial_data}")
    print(f"  Dry run: {args.dry_run}")
    
    # Define all paths early
    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"sail_{args.parcel_size}x{args.parcel_size}_rep{args.replicate}"
    
    # Output file with _klam suffix to distinguish from old surrogate-based files
    output_file = output_dir / f"sail_{args.parcel_size}x{args.parcel_size}_rep{args.replicate}_klam.npz"
    
    # Spatial data file (separate from main data for flexibility)
    spatial_file = output_dir / f"sail_{args.parcel_size}x{args.parcel_size}_rep{args.replicate}_spatial.npz"
    
    # Old surrogate-based .npz file (from previous runs)
    old_npz_file = output_dir / f"sail_{args.parcel_size}x{args.parcel_size}_rep{args.replicate}.npz"
    
    # Archive path (generated by SAIL with timestamp prefix)
    # Pattern: YYYY-MM-DD_HH-MM-SSFinalQD_archive.pkl
    archive_paths = list(run_dir.glob('*FinalQD_archive.pkl'))
    archive_path = archive_paths[0] if archive_paths else run_dir / "FinalQD_archive.pkl"
    
    print(f"\nPaths:")
    print(f"  Run directory: {run_dir}")
    print(f"  Old NPZ file: {old_npz_file}")
    print(f"  Archive pattern: {run_dir}/*FinalQD_archive.pkl")
    if archive_paths:
        print(f"  Archive found: {archive_path.name}")
    print(f"  Output file: {output_file}")
    if args.collect_spatial_data:
        print(f"  Spatial file: {spatial_file}")
    
    # ========================================================================
    # STAGE 1: Check if final _klam.npz output exists
    # ========================================================================
    print("\n" + "-"*70)
    print("STAGE 1: Checking existing output files")
    print("-"*70)
    
    klam_file_exists = output_file.exists()
    klam_file_valid = False
    klam_file_elites = 0
    
    if klam_file_exists:
        try:
            existing = np.load(output_file)
            klam_file_elites = len(existing['objectives'])
            # Verify it has real KLAM objectives (not surrogate)
            if 'evaluation_method' in existing:
                klam_file_valid = existing['evaluation_method'] == 'klam_21'
            else:
                # Old format - assume valid if objectives exist
                klam_file_valid = klam_file_elites > 0
            print(f"  ✓ Found existing _klam.npz with {klam_file_elites} elites")
        except Exception as e:
            print(f"  ⚠ Existing _klam.npz is corrupted: {e}")
    else:
        print(f"  → No existing _klam.npz found")
    
    # ========================================================================
    # STAGE 2: Check for existing data sources (old .npz or archive.pkl)
    # ========================================================================
    print("\n" + "-"*70)
    print("STAGE 2: Checking for existing SAIL data")
    print("-"*70)
    
    # Priority: old .npz file > archive.pkl > run SAIL
    old_npz_exists = old_npz_file.exists()
    old_npz_valid = False
    old_npz_elites = 0
    
    if old_npz_exists:
        try:
            test_data = np.load(old_npz_file)
            old_npz_elites = len(test_data['objectives'])
            old_npz_valid = old_npz_elites > 0 and 'genomes' in test_data
            if old_npz_valid:
                print(f"  ✓ Found old .npz with {old_npz_elites} elites (surrogate-evaluated)")
        except Exception as e:
            print(f"  ⚠ Old .npz is corrupted: {e}")
    else:
        print(f"  → No old .npz found")
    
    archive_exists = archive_path.exists()
    archive_valid = False
    archive_elites = 0
    archive_complete = False
    archive_generation = 0
    sail_phase_complete = False
    finalqd_generation = 0
    
    if not old_npz_valid and archive_exists:
        try:
            # Check if SAIL optimization completed (both phases)
            archive_complete, archive_generation, sail_phase_complete, finalqd_generation = check_sail_completion(
                run_dir, args.num_generations
            )
            
            # Load archive to get elite count
            archive = load_archive_from_pickle(archive_path)
            archive_elites = len(archive.data()['objective'])
            archive_valid = True
            
            print(f"  ✓ Archive exists: {archive_path}")
            print(f"    Elites: {archive_elites:,}")
            print(f"    SAIL phase: {'✓ COMPLETE' if sail_phase_complete else f'{archive_generation if not sail_phase_complete else args.num_generations:,} / {args.num_generations:,}'}")
            if sail_phase_complete:
                print(f"    FinalQD phase: {finalqd_generation:,} / {args.num_generations:,}")
                print(f"    Total progress: {archive_generation:,} / {2 * args.num_generations:,}")
            
            if archive_complete:
                print(f"    Status: ✓ COMPLETE (both phases finished)")
            elif sail_phase_complete:
                print(f"    Status: ⚠ FinalQD INCOMPLETE (can resume from FinalQD gen {finalqd_generation})")
            else:
                print(f"    Status: ⚠ SAIL INCOMPLETE (can resume from gen {archive_generation})")
                
        except Exception as e:
            print(f"  ⚠ Archive exists but invalid: {e}")
            import traceback
            traceback.print_exc()
    elif not old_npz_valid:
        print(f"  → No archive found")
    
    # Determine data source and whether SAIL needs to run/resume
    has_existing_data = old_npz_valid or archive_valid
    
    if old_npz_valid:
        data_source = "old_npz"
        data_source_elites = old_npz_elites
        need_sail = False
        resume_sail = False
    elif archive_valid and archive_complete:
        data_source = "archive_complete"
        data_source_elites = archive_elites
        need_sail = False
        resume_sail = False
    elif archive_valid and not archive_complete:
        data_source = "archive_incomplete"
        data_source_elites = archive_elites
        need_sail = True
        resume_sail = True
    else:
        data_source = "sail"
        data_source_elites = 0
        need_sail = True
        resume_sail = False
    
    # ========================================================================
    # DRY RUN MODE
    # ========================================================================
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE - No simulations will be executed")
        print("="*70)
        
        print("\nDecision Logic:")
        print("-" * 70)
        
        if klam_file_valid:
            print(f"✓ SKIP: Final output file exists and is valid")
            print(f"  File: {output_file}")
            print(f"  Elites: {klam_file_elites:,}")
            print(f"  Action: Would skip entirely (use --force to regenerate)")
        else:
            if klam_file_exists:
                print(f"⚠ Final output file exists but is INVALID")
                print(f"  File: {output_file}")
            else:
                print(f"✗ Final output file does not exist")
                print(f"  File: {output_file}")
            
            print(f"\n  Data Source: {data_source}")
            
            if data_source == "old_npz":
                print(f"    → Would load {old_npz_elites:,} genomes from: {old_npz_file}")
                print(f"    → Would SKIP SAIL optimization (genomes exist)")
            elif data_source == "archive_complete":
                print(f"    → Would load {archive_elites:,} genomes from: {archive_path}")
                print(f"    → Would SKIP SAIL optimization (both phases complete)")
            elif data_source == "archive_incomplete":
                print(f"    → Would load {archive_elites:,} genomes from: {archive_path}")
                if sail_phase_complete:
                    print(f"    → Would RESUME FinalQD phase from generation {finalqd_generation:,}")
                    print(f"      Remaining FinalQD: {args.num_generations - finalqd_generation:,} generations")
                else:
                    print(f"    → Would RESUME SAIL phase from generation {archive_generation:,}")
                    print(f"      Remaining SAIL: {args.num_generations - archive_generation:,} generations")
                    print(f"      Then FinalQD: {args.num_generations:,} generations")
                print(f"      Output: {run_dir}")
            else:
                print(f"    → Would RUN SAIL optimization from scratch")
                print(f"      SAIL Generations: {args.num_generations:,}")
                print(f"      FinalQD Generations: {args.num_generations:,}")
                print(f"      Output: {run_dir}")
            
            print(f"\n  KLAM Evaluation:")
            if data_source_elites > 0:
                print(f"    → Would evaluate {data_source_elites:,} elites with REAL KLAM_21")
            else:
                print(f"    → Would evaluate all SAIL archive elites with REAL KLAM_21")
            print(f"      Workers: {args.num_workers or 'all CPUs'}")
            print(f"      Batch size: {args.batch_size}")
            
            print(f"\n  Output:")
            print(f"    → Would save to: {output_file}")
            if args.collect_spatial_data:
                print(f"    → Would save spatial data to: {spatial_file}")
        
        print("\n" + "="*70)
        print("DRY RUN COMPLETE - No files were modified")
        print("="*70)
        return
    
    # ========================================================================
    # NORMAL MODE: Apply skip logic
    # ========================================================================
    if args.skip_existing and not args.force and klam_file_valid:
        print(f"\n✓ SKIPPING: Valid _klam.npz already exists with {klam_file_elites} elite solutions")
        print(f"  File: {output_file}")
        print(f"  Use --force to regenerate")
        sys.exit(0)  # Success - nothing to do
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load configs
    with open(project_root / "domain_description/cfg.yml") as f:
        config_environment = yaml.safe_load(f)
    
    with open(project_root / "optimization/cfg.yml") as f:
        config_optimization = yaml.safe_load(f)
    
    with open(project_root / "encodings/parametric/cfg.yml") as f:
        config_encoding = yaml.safe_load(f)
    
    # Override configs for this run
    config_environment['evaluation_method'] = 'klam'
    config_environment['length_design'] = design_cells
    
    config_optimization['num_generations'] = args.num_generations
    
    config_encoding['parcel_width_m'] = float(args.parcel_size)
    config_encoding['parcel_height_m'] = float(args.parcel_size)
    config_encoding['length_design'] = design_cells
    
    # Create encoding template
    solution_template = ParametricEncoding(config=config_encoding)
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STAGE 3: Load data or run/resume SAIL
    # ========================================================================
    data = None
    
    if old_npz_valid:
        # Load from existing .npz file (fastest path)
        print("\n" + "-"*70)
        print("STAGE 3: Loading genomes from existing .npz file (skipping SAIL)")
        print("-"*70)
        
        data = load_data_from_npz(old_npz_file)
        if data is None:
            print("ERROR: Failed to load data from old .npz file!")
            sys.exit(1)
        
        print(f"  Loaded {len(data['genomes']):,} elites from {old_npz_file.name}")
        print(f"  Surrogate objective range: [{data['surrogate_objectives'].min():.4f}, {data['surrogate_objectives'].max():.4f}]")
        print(f"  NOTE: These objectives are SURROGATE predictions, not real KLAM_21!")
        
    elif archive_valid and archive_complete:
        # Load from completed archive (skip SAIL)
        print("\n" + "-"*70)
        print("STAGE 3: Loading genomes from completed archive (skipping SAIL)")
        print("-"*70)
        print(f"  Both phases complete: SAIL {args.num_generations:,} + FinalQD {args.num_generations:,}")
        
        archive = load_archive_from_pickle(archive_path)
        data = extract_archive_data(archive, float(args.parcel_size))
        
        if data is None:
            print("ERROR: No elites in archive!")
            sys.exit(1)
        
        print(f"  Loaded {len(data['genomes']):,} elites from archive")
        print(f"  Surrogate objective range: [{data['surrogate_objectives'].min():.4f}, {data['surrogate_objectives'].max():.4f}]")
        print(f"  NOTE: These objectives are SURROGATE predictions, not real KLAM_21!")
        
    elif archive_valid and not archive_complete:
        # Resume SAIL/FinalQD from incomplete archive
        print("\n" + "-"*70)
        if sail_phase_complete:
            print("STAGE 3: Resuming FinalQD phase from incomplete archive")
            print("-"*70)
            print(f"  SAIL phase: ✓ COMPLETE ({args.num_generations:,} generations)")
            print(f"  FinalQD phase: {finalqd_generation:,}/{args.num_generations:,} (INCOMPLETE)")
            print(f"  Total progress: {archive_generation:,}/{2 * args.num_generations:,}")
            print(f"  Remaining FinalQD: {args.num_generations - finalqd_generation:,} generations")
        else:
            print("STAGE 3: Resuming SAIL phase from incomplete archive")
            print("-"*70)
            print(f"  SAIL phase: {archive_generation:,}/{args.num_generations:,} (INCOMPLETE)")
            print(f"  Remaining SAIL: {args.num_generations - archive_generation:,} generations")
            print(f"  Then FinalQD: {args.num_generations:,} generations")
        
        print(f"  Archive path: {archive_path}")
        
        # Determine which phase's stats to load
        if sail_phase_complete:
            # Load FinalQD stats for resume
            finalqd_stats_path = list(run_dir.glob('*FinalQD_stats.pkl'))
            if finalqd_stats_path:
                archive_stats_path = finalqd_stats_path[0]
            else:
                # FinalQD just started, no stats yet - will start from 0
                archive_stats_path = None
        else:
            # Load SAIL stats for resume
            sail_stats_path = list(run_dir.glob('*SAIL_stats.pkl'))
            if sail_stats_path:
                archive_stats_path = sail_stats_path[0]
            else:
                archive_stats_path = None
        
        if archive_stats_path:
            print(f"  Stats path: {archive_stats_path}")
        else:
            print(f"  Stats path: None (will start fresh)")
        
        print(f"  This will continue optimization where it left off")
        
        # Load existing archive to resume from
        print("  Loading existing archive...")
        existing_archive = load_archive_from_pickle(archive_path)
        print(f"  Current archive has {len(existing_archive.data()['objective']):,} elites")
        
        # Load existing stats
        existing_stats = None
        if archive_stats_path and archive_stats_path.exists():
            print("  Loading existing stats...")
            with open(archive_stats_path, 'rb') as f:
                existing_stats = pickle.load(f)
            print(f"  Stats contains {len(existing_stats):,} entries (generation {len(existing_stats)-1})")
        else:
            print("  No stats file to load, will start fresh")
        
        # Load GP training data (actual KLAM evaluations) - always from SAIL phase
        # Try multiple possible paths
        gp_data_paths = list(run_dir.glob('*SAIL_gp_data.pkl')) + list(run_dir.glob('*FinalQD_gp_data.pkl'))
        gp_training_X, gp_training_y = None, None
        
        for gp_data_path in gp_data_paths:
            if gp_data_path.exists():
                print(f"  Loading GP training data from {gp_data_path.name}...")
                with open(gp_data_path, 'rb') as f:
                    gp_data = pickle.load(f)
                gp_training_X = gp_data['X']
                gp_training_y = gp_data['y']
                print(f"  Loaded {len(gp_training_y):,} KLAM-evaluated samples for GP training")
                break
        
        if gp_training_X is None:
            print(f"  WARNING: GP training data file not found!")
            print(f"  Searched for: *SAIL_gp_data.pkl or *FinalQD_gp_data.pkl in {run_dir}")
            print(f"  This is expected for old runs. Will fail during resume - GP data is required!")
        
        # Update config to resume from current generation
        config_optimization_resume = config_optimization.copy()
        config_optimization_resume['num_generations'] = args.num_generations
        
        start_time = datetime.now()
        
        archive, labels, sail_output_path = run_sail_optimization(
            config_environment=config_environment,
            solution_template=solution_template,
            result_path=str(run_dir),
            config_optimization=config_optimization_resume,
            config_encoding=config_encoding,
            run_parallel=True,
            debug=False,
            resume_archive=existing_archive,  # Pass loaded archive
            resume_stats=existing_stats,      # Pass loaded stats (for correct phase)
            resume_generation=archive_generation,  # Pass TOTAL generation count (SAIL + FinalQD)
            resume_gp_data=(gp_training_X, gp_training_y) if gp_training_X is not None else None  # Pass GP training data
        )
        
        elapsed = datetime.now() - start_time
        print(f"\n  SAIL resumed and completed in {elapsed}")
        print(f"  Archive coverage: {archive.stats.coverage:.2%}")
        print(f"  Archive QD score: {archive.stats.qd_score:.4f}")
        
        # Extract data from archive
        print("\n  Extracting elite solutions...")
        data = extract_archive_data(archive, float(args.parcel_size))
        
        if data is None:
            print("ERROR: No elites in archive!")
            sys.exit(1)
        
        print(f"  Final archive elites: {len(data['genomes']):,}")
        
    else:
        # Run SAIL from scratch
        print("\n" + "-"*70)
        print("STAGE 3: Running SAIL optimization")
        print("-"*70)
        
        start_time = datetime.now()
        
        archive, labels, sail_output_path = run_sail_optimization(
            config_environment=config_environment,
            solution_template=solution_template,
            result_path=str(run_dir),
            config_optimization=config_optimization,
            config_encoding=config_encoding,
            run_parallel=True,
            debug=False
        )
        
        elapsed = datetime.now() - start_time
        print(f"\n  SAIL completed in {elapsed}")
        print(f"  Archive coverage: {archive.stats.coverage:.2%}")
        print(f"  Archive QD score: {archive.stats.qd_score:.4f}")
        
        # Extract data from archive
        print("\n  Extracting elite solutions...")
        data = extract_archive_data(archive, float(args.parcel_size))
        
        if data is None:
            print("ERROR: No elites in archive!")
            sys.exit(1)
        
        print(f"  Extracted {len(data['genomes']):,} elites")
    
    n_elites = len(data['genomes'])
    
    # ========================================================================
    # STAGE 4: Evaluate ALL elites with real KLAM_21
    # ========================================================================
    print("\n" + "-"*70)
    print("STAGE 4: Evaluating elites with REAL KLAM_21 simulation")
    print("-"*70)
    
    start_time = datetime.now()
    
    objectives, spatial_data_list = evaluate_elites_with_klam(
        data['genomes'],
        config_environment,
        config_encoding,
        solution_template,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collect_spatial_data=args.collect_spatial_data
    )
    
    elapsed = datetime.now() - start_time
    print(f"  Total KLAM evaluation time: {elapsed}")
    
    # Statistics
    valid_mask = ~np.isnan(objectives)
    n_valid = np.sum(valid_mask)
    
    if n_valid > 0:
        print(f"  Real KLAM_21 objective range: [{objectives[valid_mask].min():.4f}, {objectives[valid_mask].max():.4f}]")
        print(f"  Mean objective: {objectives[valid_mask].mean():.4f}")
    
    # Compare with surrogate predictions
    if n_valid > 0:
        surrogate = data['surrogate_objectives'][valid_mask]
        real = objectives[valid_mask]
        correlation = np.corrcoef(surrogate, real)[0, 1]
        print(f"  Surrogate vs Real correlation: {correlation:.4f}")
    
    # ========================================================================
    # STAGE 5: Save to NPZ with _klam suffix
    # ========================================================================
    print("\n" + "-"*70)
    print("STAGE 5: Saving to NPZ file")
    print("-"*70)
    
    np.savez_compressed(
        output_file,
        # Main data
        genomes=data['genomes'],
        widths=data['widths'],
        heights=data['heights'],
        objectives=objectives.astype(np.float32),  # Real KLAM_21 fitness!
        features=data['features'],  # From archive (no recalculation needed)
        phenotypes=data['phenotypes'],
        # Metadata
        parcel_size=args.parcel_size,
        replicate=args.replicate,
        num_generations=args.num_generations,
        seed=args.seed,
        evaluation_method='klam_21',  # Mark as real KLAM evaluation
        # Keep surrogate predictions for analysis
        surrogate_objectives=data['surrogate_objectives']
    )
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved to: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Save spatial data if collected
    if args.collect_spatial_data and spatial_data_list is not None:
        print("\n  Saving spatial data...")
        save_spatial_data(
            output_path=spatial_file,
            spatial_data_list=spatial_data_list,
            parcel_size=args.parcel_size,
            replicate=args.replicate
        )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("✓ SAIL DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Parcel size: {args.parcel_size}×{args.parcel_size} m")
    print(f"  Replicate: {args.replicate}")
    print(f"  Total elites: {n_elites:,}")
    print(f"  Valid KLAM evaluations: {n_valid:,} ({100*n_valid/n_elites:.1f}%)")
    print(f"  Failed evaluations (NaN): {n_elites - n_valid:,}")
    print(f"  Output file: {output_file}")
    if args.collect_spatial_data:
        print(f"  Spatial file: {spatial_file}")
    print(f"  NOTE: objectives are REAL KLAM_21 values, not surrogate predictions!")


if __name__ == "__main__":
    main()
