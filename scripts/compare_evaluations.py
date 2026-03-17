# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Use proper package imports
from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from domain_description.evaluation import eval as eval_floodfill
from domain_description.evaluation import init_environment as init_floodfill
from domain_description.evaluation_klam import eval as eval_klam
from domain_description.evaluation_klam import init_environment as init_klam

# Global variables for multiprocessing (pickle workaround)
_config_ff = None
_config_klam = None
_config_encoding = None
_solution_template = None


def init_worker(config_ff, config_klam, config_encoding):
    """Initialize worker process with configs."""
    global _config_ff, _config_klam, _config_encoding, _solution_template
    _config_ff = config_ff
    _config_klam = config_klam
    _config_encoding = config_encoding
    _solution_template = ParametricEncoding()


def evaluate_solution(sol):
    """Evaluate a single solution with both methods."""
    try:
        ff_result, _ = eval_floodfill(sol, _config_ff, _config_encoding, _solution_template, use_surrogate=False)
        klam_result, _ = eval_klam(sol, _config_klam, _config_encoding, _solution_template, use_surrogate=False)
        return ff_result[0], klam_result[0]
    except Exception as e:
        print(f"Error evaluating solution: {e}")
        return np.nan, np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Compare flood-fill and KLAM_21 evaluation on an archive of solutions."
    )
    parser.add_argument(
        '--archive_path', 
        required=True, 
        type=str, 
        help="Path to the archive.pkl file containing solutions."
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=100,
        help="Number of solutions to sample (default: 100). Use -1 for all."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)."
    )
    args = parser.parse_args()

    print(f"Loading solutions from: {args.archive_path}")
    if not os.path.exists(args.archive_path):
        print(f"Error: Archive file not found at {args.archive_path}")
        return

    with open(args.archive_path, 'rb') as f:
        archive = pickle.load(f)

    # Handle different PyRibs API versions
    try:
        # PyRibs >= 0.6.0: use data() method
        if hasattr(archive, 'data'):
            data = archive.data()
            solutions = data['solution']
        # PyRibs >= 0.5.0: use as_pandas() 
        elif hasattr(archive, 'as_pandas'):
            solutions = archive.as_pandas().filter(regex='solution_').to_numpy()
        # Fallback: direct attribute access
        else:
            solutions = np.array([elite.solution for elite in archive])
    except Exception as e:
        print(f"Error extracting solutions: {e}")
        print(f"Archive type: {type(archive)}")
        print(f"Archive attributes: {dir(archive)}")
        return
    
    print(f"Found {len(solutions)} solutions in the archive.")

    # Sample solutions if requested
    if args.sample_size > 0 and args.sample_size < len(solutions):
        np.random.seed(args.seed)
        sample_indices = np.random.choice(len(solutions), args.sample_size, replace=False)
        solutions = solutions[sample_indices]
        print(f"Sampled {len(solutions)} solutions (seed={args.seed})")

    solution_template = ParametricEncoding()
    with open("encodings/parametric/cfg.yml") as f:
        config_encoding = yaml.safe_load(f)
    
    with open("domain_description/cfg.yml") as f:
        base_config = yaml.safe_load(f)
    
    # Initialize both environment configs correctly
    config_ff = init_floodfill(base_config.copy())
    config_klam = init_klam(base_config.copy())

    floodfill_fitness = []
    klam_fitness = []

    num_workers = args.num_workers or cpu_count()
    
    print(f"Evaluating {len(solutions)} solutions with both methods...")
    print(f"Using {num_workers} parallel workers")
    print(f"Estimated time: ~{len(solutions) * 5 / num_workers / 60:.1f} hours (at 5 min/KLAM eval)")
    
    if num_workers > 1:
        # Parallel evaluation
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(config_ff, config_klam, config_encoding)
        ) as pool:
            results = list(tqdm(
                pool.imap(evaluate_solution, solutions),
                total=len(solutions),
                desc="Evaluating Solutions"
            ))
        
        floodfill_fitness = np.array([r[0] for r in results])
        klam_fitness = np.array([r[1] for r in results])
    else:
        # Sequential evaluation (for debugging)
        for sol in tqdm(solutions, desc="Evaluating Solutions"):
            ff_result_array, _ = eval_floodfill(sol, config_ff, config_encoding, solution_template, use_surrogate=False)
            floodfill_fitness.append(ff_result_array[0])

            klam_result_array, _ = eval_klam(sol, config_klam, config_encoding, solution_template, use_surrogate=False)
            klam_fitness.append(klam_result_array[0])
        
        floodfill_fitness = np.array(floodfill_fitness)
        klam_fitness = np.array(klam_fitness)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(floodfill_fitness) | np.isnan(klam_fitness))
    floodfill_fitness = floodfill_fitness[valid_mask]
    klam_fitness = klam_fitness[valid_mask]
    print(f"Valid evaluations: {len(floodfill_fitness)} / {len(solutions)}")

    # Save results to NPZ for later analysis
    output_dir = os.path.dirname(args.archive_path) or '.'
    npz_path = os.path.join(output_dir, 'comparison_results.npz')
    np.savez(npz_path,
             floodfill_fitness=floodfill_fitness,
             klam_fitness=klam_fitness,
             sample_size=len(floodfill_fitness),
             seed=args.seed)
    print(f"Results saved to: {npz_path}")

    plt.figure(figsize=(10, 8))
    plt.scatter(floodfill_fitness, klam_fitness, alpha=0.6, edgecolors='k')
    correlation = np.corrcoef(floodfill_fitness, klam_fitness)[0, 1]
    plt.title(f'Comparison of Fitness Functions (n={len(floodfill_fitness)})\nCorrelation Coefficient: {correlation:.4f}', fontsize=16)
    plt.xlabel('Flood-Fill Fitness (Directional Porosity)', fontsize=12)
    plt.ylabel('KLAM-21 Fitness (Wind Speed Reduction)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add regression line
    if len(floodfill_fitness) > 1:
        z = np.polyfit(floodfill_fitness, klam_fitness, 1)
        p = np.poly1d(z)
        x_line = np.linspace(floodfill_fitness.min(), floodfill_fitness.max(), 100)
        plt.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')
        plt.legend()
    
    output_filename = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison complete. Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()