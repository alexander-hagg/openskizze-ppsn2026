#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Benchmark U-Net QD Optimization Performance

Compares OLD vs NEW approaches to show speedup from eliminating double
phenotype expression in the QD evaluation loop.

This benchmark simulates the EXACT QD optimization loop from run_mapelites_offline.py
to ensure realistic timing measurements.
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp6_qd_comparison.run_mapelites_offline import UNetEvaluator, compute_features_batch
from encodings.parametric.fast_encoding import NumbaFastEncoding


def benchmark_old_approach(evaluator, genomes, parcel_sizes, fast_encoding, num_runs=10):
    """
    OLD approach: U-Net computes objectives only, then compute features separately.
    
    This simulates what would happen if U-Net evaluator didn't return features,
    requiring a separate call to compute_features_batch() which re-expresses phenotypes.
    """
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        
        # Step 1: Evaluate with U-Net (computes heightmaps internally)
        # In old code, this only returned objectives
        result = evaluator.evaluate(genomes, parcel_sizes, None)
        objectives = result['objective_predicted']
        
        # Step 2: PROBLEM - compute_features_batch() re-computes heightmaps!
        # This is wasteful since evaluator already computed them
        features_old = compute_features_batch(genomes, parcel_sizes[0], encoding=fast_encoding)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return times


def benchmark_new_approach(evaluator, genomes, parcel_sizes, num_runs=10):
    """
    NEW approach: U-Net returns features computed from already-expressed heightmaps.
    
    This is the optimized code where UNetEvaluator.evaluate() computes features
    from the heightmaps it already created for U-Net input, avoiding re-expression.
    """
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        
        # Step 1: Evaluate with U-Net (returns both objectives AND features)
        result = evaluator.evaluate(genomes, parcel_sizes, None)
        objectives = result['objective_predicted']
        features = result['features']  # Already computed, no re-expression
        
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark U-Net QD optimization performance")
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size (128 emitters × 8 solutions = 1024 per generation)')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of benchmark runs')
    parser.add_argument('--num-warmup', type=int, default=3,
                       help='Number of warmup runs')
    parser.add_argument('--unet-model', type=str,
                       default='results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth',
                       help='Path to U-Net model')
    parser.add_argument('--parcel-size', type=int, default=120,
                       choices=[60, 120, 240],
                       help='Parcel size in meters')
    parser.add_argument('--output', type=str,
                       default='results/exp6_qd_comparison/benchmark_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("U-NET QD OPTIMIZATION PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size} (realistic: 128 emitters × 8 solutions)")
    print(f"  Num runs: {args.num_runs}")
    print(f"  Num warmup: {args.num_warmup}")
    print(f"  Random seed: {args.seed}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load U-Net model
    print(f"\nLoading U-Net model...")
    unet_path = Path(args.unet_model)
    if not unet_path.exists():
        print(f"ERROR: U-Net model not found at {unet_path}")
        sys.exit(1)
    
    evaluator = UNetEvaluator(unet_path, device)
    print("✓ U-Net model loaded")
    
    # Initialize fast encoding
    print("\nInitializing NumbaFastEncoding...")
    fast_encoding = NumbaFastEncoding(parcel_size=args.parcel_size)
    print("✓ NumbaFastEncoding initialized")
    
    # Generate test data (realistic QD batch)
    print(f"\nGenerating test data ({args.batch_size} genomes)...")
    genomes = np.random.randn(args.batch_size, 60).astype(np.float32)
    parcel_sizes = np.full(args.batch_size, args.parcel_size, dtype=np.float32)
    print("✓ Test data generated")
    
    # Warmup runs
    print(f"\nWarmup ({args.num_warmup} runs)...")
    for i in range(args.num_warmup):
        result = evaluator.evaluate(genomes, parcel_sizes, None)
        if i == 0:
            print(f"  First run: objectives shape={result['objective_predicted'].shape}, "
                  f"features shape={result['features'].shape}")
    print("✓ Warmup complete")
    
    # Benchmark OLD approach (double phenotype expression)
    print(f"\n{'='*80}")
    print("BENCHMARK 1: OLD APPROACH (double phenotype expression)")
    print(f"{'='*80}")
    print("Description: U-Net evaluator returns objectives only,")
    print("             then compute_features_batch() re-expresses phenotypes")
    print(f"\nRunning {args.num_runs} iterations...")
    
    times_old = benchmark_old_approach(evaluator, genomes, parcel_sizes, fast_encoding, args.num_runs)
    
    print(f"\nResults (OLD approach):")
    for i, t in enumerate(times_old, 1):
        print(f"  Run {i:2d}: {t:7.2f} ms")
    print(f"  Mean:   {np.mean(times_old):7.2f} ± {np.std(times_old):5.2f} ms")
    print(f"  Median: {np.median(times_old):7.2f} ms")
    print(f"  Min:    {np.min(times_old):7.2f} ms")
    print(f"  Max:    {np.max(times_old):7.2f} ms")
    
    # Benchmark NEW approach (features from evaluator)
    print(f"\n{'='*80}")
    print("BENCHMARK 2: NEW APPROACH (features from evaluator)")
    print(f"{'='*80}")
    print("Description: U-Net evaluator returns both objectives AND features")
    print("             computed from already-expressed heightmaps")
    print(f"\nRunning {args.num_runs} iterations...")
    
    times_new = benchmark_new_approach(evaluator, genomes, parcel_sizes, args.num_runs)
    
    print(f"\nResults (NEW approach):")
    for i, t in enumerate(times_new, 1):
        print(f"  Run {i:2d}: {t:7.2f} ms")
    print(f"  Mean:   {np.mean(times_new):7.2f} ± {np.std(times_new):5.2f} ms")
    print(f"  Median: {np.median(times_new):7.2f} ms")
    print(f"  Min:    {np.min(times_new):7.2f} ms")
    print(f"  Max:    {np.max(times_new):7.2f} ms")
    
    # Compute speedup
    print(f"\n{'='*80}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*80}")
    
    mean_old = np.mean(times_old)
    mean_new = np.mean(times_new)
    speedup = mean_old / mean_new
    time_saved = mean_old - mean_new
    
    print(f"  OLD approach: {mean_old:.2f} ms per generation")
    print(f"  NEW approach: {mean_new:.2f} ms per generation")
    print(f"  Speedup:      {speedup:.2f}×")
    print(f"  Time saved:   {time_saved:.2f} ms per generation")
    
    # Extrapolate to full optimization
    num_generations = 10000
    total_time_saved_min = (time_saved * num_generations) / 1000 / 60
    
    print(f"\n  For {num_generations} generations:")
    print(f"    OLD: {mean_old * num_generations / 1000 / 60:.1f} minutes")
    print(f"    NEW: {mean_new * num_generations / 1000 / 60:.1f} minutes")
    print(f"    SAVED: {total_time_saved_min:.1f} minutes ({total_time_saved_min/60:.1f} hours)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': {
            'batch_size': args.batch_size,
            'num_runs': args.num_runs,
            'num_warmup': args.num_warmup,
            'seed': args.seed,
            'device': str(device),
            'unet_model': str(args.unet_model),
        },
        'old_approach': {
            'times_ms': times_old,
            'mean_ms': float(np.mean(times_old)),
            'std_ms': float(np.std(times_old)),
            'median_ms': float(np.median(times_old)),
            'min_ms': float(np.min(times_old)),
            'max_ms': float(np.max(times_old)),
        },
        'new_approach': {
            'times_ms': times_new,
            'mean_ms': float(np.mean(times_new)),
            'std_ms': float(np.std(times_new)),
            'median_ms': float(np.median(times_new)),
            'min_ms': float(np.min(times_new)),
            'max_ms': float(np.max(times_new)),
        },
        'speedup': {
            'factor': float(speedup),
            'time_saved_ms_per_generation': float(time_saved),
            'time_saved_minutes_10k_gens': float(total_time_saved_min),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
