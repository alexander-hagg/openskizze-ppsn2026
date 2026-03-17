#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Detailed Profiling of U-Net QD Evaluation Bottlenecks

Breaks down where time is spent in the evaluation pipeline to identify
the actual bottlenecks.
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

from experiments.exp6_qd_comparison.run_mapelites_offline import UNetEvaluator, optimized_construct_domain_grids
from encodings.parametric.fast_encoding import NumbaFastEncoding, numba_calculate_features


class ProfilingUNetEvaluator(UNetEvaluator):
    """U-Net evaluator with detailed timing breakdowns."""
    
    def evaluate_with_timing(self, genomes: np.ndarray, parcel_sizes: np.ndarray):
        """Evaluate with detailed timing for each step."""
        timings = {}
        
        # Expected dimensions
        expected_h = self.model.config.input_height  # 66
        expected_w = self.model.config.input_width   # 94
        parcel_size_cells = 9
        
        # Step 1: Express batch (genome → heightmap)
        start = time.perf_counter()
        heightmaps = self.fast_encoding.express_batch(genomes)
        timings['phenotype_expression'] = (time.perf_counter() - start) * 1000
        
        # Step 2: Construct domain grids (heightmap → U-Net input)
        start = time.perf_counter()
        terrain, buildings, landuse = optimized_construct_domain_grids(
            heightmaps, 
            parcel_size_cells,
            grid_h=expected_h,
            grid_w=expected_w
        )
        timings['domain_construction'] = (time.perf_counter() - start) * 1000
        
        # Step 3: Normalize inputs
        start = time.perf_counter()
        terrain_norm = (terrain - self.terrain_mean) / self.terrain_std
        buildings_norm = (buildings - self.buildings_mean) / self.buildings_std
        landuse_norm = (landuse - self.landuse_mean) / self.landuse_std
        X = np.stack([terrain_norm, buildings_norm, landuse_norm], axis=1)
        timings['input_normalization'] = (time.perf_counter() - start) * 1000
        
        # Step 4: CPU→GPU transfer
        start = time.perf_counter()
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        timings['cpu_to_gpu_transfer'] = (time.perf_counter() - start) * 1000
        
        # Step 5: U-Net inference
        start = time.perf_counter()
        with torch.no_grad():
            Y_pred = self.model(X_torch)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        timings['unet_inference'] = (time.perf_counter() - start) * 1000
        
        # Step 6: GPU→CPU transfer
        start = time.perf_counter()
        Y_pred = Y_pred.cpu().numpy()
        timings['gpu_to_cpu_transfer'] = (time.perf_counter() - start) * 1000
        
        # Step 7: Denormalize outputs
        start = time.perf_counter()
        output_vars = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']
        predictions = {}
        for i, var in enumerate(output_vars):
            predictions[var] = Y_pred[:, i, :, :] * self.output_stds[var] + self.output_means[var]
        timings['output_denormalization'] = (time.perf_counter() - start) * 1000
        
        # Step 8: Compute cold air flux
        start = time.perf_counter()
        Ex = predictions['Ex']
        uq = predictions['uq']
        vq = predictions['vq']
        uq_ms = uq / 100.0
        vq_ms = vq / 100.0
        wind_speed = np.sqrt(uq_ms**2 + vq_ms**2)
        roi_mask = np.ones((expected_h, expected_w), dtype=bool)
        objectives = np.zeros(len(Ex))
        for i in range(len(Ex)):
            Ex_roi = Ex[i][roi_mask]
            wind_roi = wind_speed[i][roi_mask]
            objectives[i] = np.mean(Ex_roi) * np.mean(wind_roi)
        timings['flux_computation'] = (time.perf_counter() - start) * 1000
        
        # Step 9: Compute features
        start = time.perf_counter()
        pixel_size = self.fast_encoding.config['xy_scale']
        features = np.zeros((len(heightmaps), 8), dtype=np.float64)
        for i in range(len(heightmaps)):
            features[i] = numba_calculate_features(heightmaps[i], pixel_size)
        timings['feature_computation'] = (time.perf_counter() - start) * 1000
        
        # Total
        timings['total'] = sum(timings.values())
        
        return objectives, features, timings


def main():
    parser = argparse.ArgumentParser(description="Profile U-Net QD evaluation bottlenecks")
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--num-runs', type=int, default=20,
                       help='Number of profiling runs')
    parser.add_argument('--unet-model', type=str,
                       default='results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth',
                       help='Path to U-Net model')
    parser.add_argument('--parcel-size', type=int, default=120,
                       choices=[60, 120, 240],
                       help='Parcel size in meters')
    parser.add_argument('--output', type=str,
                       default='results/exp6_qd_comparison/profiling_results.json',
                       help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("U-NET QD EVALUATION - DETAILED PROFILING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num runs: {args.num_runs}")
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading U-Net model...")
    unet_path = Path(args.unet_model)
    evaluator = ProfilingUNetEvaluator(unet_path, device)
    print("✓ Model loaded")
    
    # Generate test data
    genomes = np.random.randn(args.batch_size, 60).astype(np.float32)
    parcel_sizes = np.full(args.batch_size, args.parcel_size, dtype=np.float32)
    
    # Warmup
    print(f"\nWarmup (3 runs)...")
    for _ in range(3):
        _ = evaluator.evaluate_with_timing(genomes, parcel_sizes)
    print("✓ Warmup complete")
    
    # Profile
    print(f"\nProfiling ({args.num_runs} runs)...")
    all_timings = []
    
    for i in range(args.num_runs):
        _, _, timings = evaluator.evaluate_with_timing(genomes, parcel_sizes)
        all_timings.append(timings)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{args.num_runs} runs...")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN (mean ± std over {} runs)".format(args.num_runs))
    print("=" * 80)
    
    steps = [
        'phenotype_expression',
        'domain_construction',
        'input_normalization',
        'cpu_to_gpu_transfer',
        'unet_inference',
        'gpu_to_cpu_transfer',
        'output_denormalization',
        'flux_computation',
        'feature_computation',
        'total'
    ]
    
    step_names = {
        'phenotype_expression': '1. Phenotype Expression (genome → heightmap)',
        'domain_construction': '2. Domain Construction (heightmap → U-Net input grids)',
        'input_normalization': '3. Input Normalization',
        'cpu_to_gpu_transfer': '4. CPU → GPU Transfer',
        'unet_inference': '5. U-Net Inference (GPU)',
        'gpu_to_cpu_transfer': '6. GPU → CPU Transfer',
        'output_denormalization': '7. Output Denormalization',
        'flux_computation': '8. Cold Air Flux Computation',
        'feature_computation': '9. Feature Computation (Numba JIT)',
        'total': 'TOTAL'
    }
    
    results = {}
    
    for step in steps:
        times = [t[step] for t in all_timings]
        mean_time = np.mean(times)
        std_time = np.std(times)
        pct_of_total = (mean_time / np.mean([t['total'] for t in all_timings])) * 100
        
        results[step] = {
            'mean_ms': float(mean_time),
            'std_ms': float(std_time),
            'pct_of_total': float(pct_of_total)
        }
        
        if step == 'total':
            print("-" * 80)
        
        print(f"{step_names[step]}")
        print(f"  {mean_time:7.2f} ± {std_time:5.2f} ms  ({pct_of_total:5.1f}%)")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    unet_time = results['unet_inference']['mean_ms']
    encoding_time = results['phenotype_expression']['mean_ms']
    feature_time = results['feature_computation']['mean_ms']
    transfer_time = (results['cpu_to_gpu_transfer']['mean_ms'] + 
                     results['gpu_to_cpu_transfer']['mean_ms'])
    
    print(f"• U-Net inference dominates: {unet_time:.1f}ms ({results['unet_inference']['pct_of_total']:.1f}%)")
    print(f"• GPU transfers add overhead: {transfer_time:.1f}ms")
    print(f"• Phenotype expression: {encoding_time:.1f}ms ({results['phenotype_expression']['pct_of_total']:.1f}%)")
    print(f"• Feature computation: {feature_time:.1f}ms ({results['feature_computation']['pct_of_total']:.1f}%)")
    print(f"• Eliminating double expression saves: ~{encoding_time:.1f}ms (only {encoding_time/results['total']['mean_ms']*100:.1f}% of total)")
    
    print(f"\n→ The real bottleneck is U-Net inference ({unet_time:.1f}ms), not encoding")
    print(f"→ Our optimization saves {encoding_time:.1f}ms, which is the MAXIMUM possible for encoding")
    print(f"→ For larger speedups, we'd need to optimize U-Net itself or use batch processing")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': {
            'batch_size': args.batch_size,
            'num_runs': args.num_runs,
            'device': str(device),
        },
        'timings': results,
        'all_runs': all_timings
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
