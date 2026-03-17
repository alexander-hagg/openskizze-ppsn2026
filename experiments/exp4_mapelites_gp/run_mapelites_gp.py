#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
MAP-Elites with Pre-trained GP Surrogate

This script runs MAP-Elites optimization using a pre-trained GP model as the
objective function. It explores the effect of varying numbers of emitters and
batch sizes on archive quality and diversity.

Usage:
    python experiments/run_mapelites_gp.py --num-emitters 64 --batch-size 16 --num-generations 100
    python experiments/run_mapelites_gp.py --gp-model results/hyperparameterization/model_optimized_ind2500_random_rep1.pth
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

import numpy as np
import torch
import gpytorch
import yaml
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up from exp4_mapelites_gp -> experiments -> project root
sys.path.insert(0, str(project_root))

from experiments.exp3_hpo.train_gp_hpo import SVGPModel
from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)
from domain_description.evaluation_klam import calculate_planning_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GP Model Interface
# ============================================================================

class GPEvaluator:
    """Wrapper for GP model evaluation."""
    
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        logger.info(f"Loading GP model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model parameters
        inducing_key = 'variational_strategy.inducing_points'
        inducing_points = checkpoint['model_state_dict'][inducing_key]
        num_inducing = inducing_points.size(0)
        input_dim = inducing_points.size(1)
        
        logger.info(f"  Inducing points: {num_inducing}")
        logger.info(f"  Input dimension: {input_dim}")
        
        # Create model and likelihood
        self.model = SVGPModel(inducing_points.to(device), input_dim=input_dim).to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        # Load normalization parameters
        self.train_x_mean = checkpoint['train_x_mean'].to(device)
        self.train_x_std = checkpoint['train_x_std'].to(device)
        self.train_y_mean = checkpoint['train_y_mean'].to(device)
        self.train_y_std = checkpoint['train_y_std'].to(device)
        
        # Set to eval mode
        self.model.eval()
        self.likelihood.eval()
        
        logger.info("GP model loaded successfully")
    
    def evaluate(self, genomes: np.ndarray, parcel_sizes: np.ndarray, 
                 lambda_ucb: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate solutions with GP model using UCB acquisition.
        
        Args:
            genomes: (N, 60) array of genomes
            parcel_sizes: (N,) array of parcel sizes in meters
            lambda_ucb: UCB penalty coefficient (score = mean - lambda * std)
                       Set > 0 to penalize uncertain predictions, preventing
                       exploitation of GP errors. SAIL uses 1.0.
        
        Returns:
            objectives: (N,) array of predicted objectives (UCB-adjusted if lambda > 0)
            features: (N, 8) array of planning features
        """
        # Prepare inputs (genome + 2x parcel_size for width/height)
        parcel_cols = np.column_stack([parcel_sizes, parcel_sizes])
        X = np.column_stack([genomes, parcel_cols]).astype(np.float32)
        
        # Convert to torch
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Normalize
        X_norm = (X_torch - self.train_x_mean) / (self.train_x_std + 1e-6)
        
        # Predict with uncertainty
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_norm))
            pred_mean = pred.mean * self.train_y_std + self.train_y_mean
            pred_std = pred.stddev * self.train_y_std  # Scale std by output normalization
        
        # UCB acquisition: penalize uncertain predictions
        # score = mean - lambda * std (lower score for high uncertainty)
        if lambda_ucb > 0:
            objectives = (pred_mean - lambda_ucb * pred_std).cpu().numpy()
        else:
            objectives = pred_mean.cpu().numpy()
        
        # Compute features from phenotypes
        features = self._compute_features(genomes, parcel_sizes)
        
        return objectives, features
    
    def _compute_features(self, genomes: np.ndarray, parcel_sizes: np.ndarray) -> np.ndarray:
        """Compute planning features from genomes."""
        features = []
        
        # Use parametric encoding to get phenotypes
        for genome, size in zip(genomes, parcel_sizes):
            # Create encoding with config
            config_encoding = {
                'length_design': int(size / 3), 
                'max_num_buildings': 10, 
                'max_num_floors': 10,
                'xy_scale': 3.0,
                'z_scale': 3.0
            }
            encoding = ParametricEncoding(config=config_encoding)
            # Express as 2D heightmap (floors), not 3D voxels
            heightmap = encoding.express(genome, as_height_map=True)
            
            # Calculate features
            feat = calculate_planning_features(heightmap, config_encoding)
            features.append(feat)
        
        return np.array(features)


# ============================================================================
# Diversity Metrics
# ============================================================================

def compute_pairwise_distances(phenotypes: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """
    Efficiently compute pairwise Euclidean distances.
    
    Args:
        phenotypes: (N, D) array of flattened phenotypes
        batch_size: Batch size for computation to avoid memory issues
    
    Returns:
        distances: (N*(N-1)/2,) array of pairwise distances (condensed form)
    """
    from scipy.spatial.distance import pdist
    
    # For efficiency, use scipy's optimized pdist
    logger.info(f"Computing pairwise distances for {len(phenotypes)} phenotypes...")
    distances = pdist(phenotypes, metric='euclidean')
    
    return distances


def compute_diversity_metrics(archive: GridArchive, encoding_func) -> Dict:
    """
    Compute phenotypic diversity metrics from archive.
    
    Args:
        archive: PyRibs GridArchive
        encoding_func: Function to convert genome to phenotype
    
    Returns:
        Dictionary of diversity metrics
    """
    # Get all elites from archive
    data = archive.data()
    
    if len(data['solution']) == 0:
        return {
            'n_elites': 0,
            'mean_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'std_pairwise_distance': 0.0,
            'solow_polasky_diversity': 0.0,
            'effective_n_species': 0.0,
        }
    
    # Extract genomes and parcel sizes from solutions
    solutions = np.array(data['solution'])
    genomes = solutions[:, :60]
    
    # Use genome space for diversity (all same dimension: 60D)
    # This is more robust than phenotype space when parcel sizes vary
    phenotypes = genomes
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(phenotypes)
    
    # Basic statistics
    mean_dist = float(np.mean(distances))
    min_dist = float(np.min(distances)) if len(distances) > 0 else 0.0
    max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
    std_dist = float(np.std(distances))
    
    # Solow-Polasky diversity
    # SP = sum(exp(-theta * d_ij)) where theta controls sensitivity
    # Use theta = 1/mean_distance for scale-invariance
    theta = 1.0 / (mean_dist + 1e-6)
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(distances)
    sp_diversity = float(np.sum(np.exp(-theta * dist_matrix)))
    
    # Effective number of species (exp of Shannon entropy)
    # Treat each elite equally weighted
    n_elites = len(solutions)
    effective_n = float(n_elites)  # Simplified: assumes uniform distribution
    
    return {
        'n_elites': n_elites,
        'mean_pairwise_distance': mean_dist,
        'min_pairwise_distance': min_dist,
        'max_pairwise_distance': max_dist,
        'std_pairwise_distance': std_dist,
        'distance_distribution': distances,  # For histogram
        'solow_polasky_diversity': sp_diversity,
        'effective_n_species': effective_n,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_mapelites(
    gp_model_path: Path,
    num_emitters: int,
    batch_size: int,
    num_generations: int,
    parcel_size: int,
    output_dir: Path,
    seed: int = 42,
    device: Optional[torch.device] = None,
    lambda_ucb: float = 1.0,
    genome_bounds: float = 15.0
):
    """Run MAP-Elites with GP surrogate.
    
    Args:
        lambda_ucb: UCB penalty coefficient (score = mean - lambda * std).
                   Set > 0 to penalize uncertain predictions. SAIL uses 1.0.
        genome_bounds: Clip genome values to [-bounds, bounds] to prevent
                      extrapolation beyond training data. Training data has
                      99th percentile at ±14, so 15 is a reasonable default.
    """
    
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 70)
    logger.info("MAP-Elites with GP Surrogate")
    logger.info("=" * 70)
    logger.info(f"Emitters: {num_emitters}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Generations: {num_generations}")
    logger.info(f"Parcel size: {parcel_size}m")
    logger.info(f"Lambda UCB: {lambda_ucb} (penalty for GP uncertainty)")
    logger.info(f"Genome bounds: [-{genome_bounds}, {genome_bounds}]")
    logger.info(f"Total evals: {num_emitters * batch_size * num_generations:,}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load GP model
    evaluator = GPEvaluator(gp_model_path, device)
    
    # Load domain config for feature ranges
    config_path = project_root / "domain_description" / "cfg.yml"
    with open(config_path) as f:
        domain_config = yaml.safe_load(f)
    
    # Convert feat_ranges from [[mins], [maxs]] to [(min1, max1), (min2, max2), ...]
    feat_ranges_raw = domain_config['feat_ranges']
    feature_ranges = list(zip(feat_ranges_raw[0], feat_ranges_raw[1]))
    
    # Create archive
    logger.info("Creating archive...")
    logger.info(f"  8D feature space with 5 bins each = {5**8:,} cells")
    logger.info(f"  Feature ranges: {feature_ranges}")
    archive = GridArchive(
        solution_dim=62,  # 60D genome + 2D parcel size
        dims=[5] * 8,  # 8 features, 5 bins each
        ranges=feature_ranges,
        seed=seed
    )
    
    # Create emitters
    logger.info(f"Creating {num_emitters} emitters...")
    emitters = [
        GaussianEmitter(
            archive=archive,
            sigma=1.0,
            x0=np.concatenate([
                np.random.uniform(-1, 1, 60),  # Random genome
                [parcel_size, parcel_size]  # Fixed parcel size
            ]),
            batch_size=batch_size,
            seed=seed + i
        )
        for i in range(num_emitters)
    ]
    
    # Create scheduler
    scheduler = Scheduler(archive, emitters)
    
    # Setup encoding function for diversity metrics
    def encode_phenotype(genome, length_design):
        config = {'length_design': length_design, 'max_num_buildings': 10, 'max_num_floors': 10}
        encoding = ParametricEncoding(config=config)
        # Return as 2D heightmap (in floors), then flatten for distance calculation
        return encoding.express(genome, as_height_map=True).flatten()
    
    # Tracking
    history = {
        'generations': [],
        'qd_score': [],
        'coverage': [],
        'max_objective': [],
        'n_elites': [],
        'evaluations': [],
        'wall_time': [],
        'diversity_metrics': [],
    }
    
    start_time = time.time()
    total_evals = 0
    
    logger.info("\nStarting optimization...")
    logger.info("=" * 70)
    
    # Main loop
    for gen in range(num_generations):
        # Ask for solutions
        solutions = scheduler.ask()
        
        # CRITICAL: Keep parcel size fixed (prevent mutation of last 2 dims)
        # The emitter mutates all 62 dimensions, but we want parcel size constant
        solutions[:, 60:] = parcel_size
        
        # Clip genomes to prevent extrapolation beyond training data
        # Training data has 99th percentile at ±14, so ±15 covers ~99%+ of valid range
        if genome_bounds > 0:
            solutions[:, :60] = np.clip(solutions[:, :60], -genome_bounds, genome_bounds)
        
        # Extract genomes and parcel sizes
        genomes = solutions[:, :60]
        parcel_sizes = solutions[:, 60]
        
        # Evaluate with GP using UCB acquisition
        # score = mean - lambda * std (penalizes uncertain predictions)
        objectives, features = evaluator.evaluate(genomes, parcel_sizes, lambda_ucb=lambda_ucb)
        
        # Tell results
        scheduler.tell(objectives, features)
        
        total_evals += len(solutions)
        
        # Log progress
        if gen % 1000 == 0 or gen == num_generations - 1:
            elapsed = time.time() - start_time
            
            # Archive stats
            stats = archive.stats
            qd_score = stats.qd_score
            coverage = stats.coverage
            max_obj = stats.obj_max
            
            logger.info(
                f"Gen {gen:4d} | Evals: {total_evals:7d} | "
                f"QD: {qd_score:8.1f} | Cov: {coverage:6.2%} | "
                f"Max: {max_obj:7.2f} | Time: {elapsed:6.1f}s"
            )
            
            # Compute diversity metrics
            logger.info("  Computing diversity metrics...")
            diversity = compute_diversity_metrics(archive, encode_phenotype)
            logger.info(
                f"    Elites: {diversity['n_elites']} | "
                f"Mean dist: {diversity['mean_pairwise_distance']:.2f} | "
                f"SP div: {diversity['solow_polasky_diversity']:.2f}"
            )
            
            # Record history
            history['generations'].append(gen)
            history['qd_score'].append(qd_score)
            history['coverage'].append(coverage)
            history['max_objective'].append(max_obj)
            history['n_elites'].append(diversity['n_elites'])
            history['evaluations'].append(total_evals)
            history['wall_time'].append(elapsed)
            history['diversity_metrics'].append(diversity)
            
            # Save archive snapshot
            archive_path = output_dir / f"archive_gen{gen:04d}.pkl"
            with open(archive_path, 'wb') as f:
                pickle.dump(archive, f)
    
    # Final stats
    elapsed = time.time() - start_time
    throughput = total_evals / elapsed
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Optimization Complete")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {total_evals:,}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Throughput: {throughput:.1f} evals/s")
    logger.info(f"Final QD score: {history['qd_score'][-1]:.1f}")
    logger.info(f"Final coverage: {history['coverage'][-1]:.2%}")
    logger.info(f"Final max objective: {history['max_objective'][-1]:.2f}")
    logger.info(f"Final n_elites: {history['n_elites'][-1]}")
    logger.info("")
    
    # Save final archive
    final_archive_path = output_dir / "archive_final.pkl"
    with open(final_archive_path, 'wb') as f:
        pickle.dump(archive, f)
    logger.info(f"Saved final archive to {final_archive_path}")
    
    # Save history
    history_path = output_dir / "history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    logger.info(f"Saved history to {history_path}")
    
    # Save config
    config = {
        'gp_model_path': str(gp_model_path),
        'num_emitters': num_emitters,
        'batch_size': batch_size,
        'num_generations': num_generations,
        'parcel_size': parcel_size,
        'lambda_ucb': lambda_ucb,
        'genome_bounds': genome_bounds,
        'seed': seed,
        'total_evaluations': total_evals,
        'total_time': elapsed,
        'throughput': throughput,
        'final_qd_score': history['qd_score'][-1],
        'final_coverage': history['coverage'][-1],
        'final_max_objective': history['max_objective'][-1],
        'final_n_elites': history['n_elites'][-1],
    }
    
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {config_path}")
    
    return archive, history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MAP-Elites with pre-trained GP surrogate"
    )
    parser.add_argument(
        "--gp-model",
        type=str,
        default="results/hyperparameterization/model_optimized_ind2500_random_rep1.pth",
        help="Path to trained GP model"
    )
    parser.add_argument(
        "--num-emitters",
        type=int,
        required=True,
        help="Number of emitters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size per emitter"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5000,
        help="Number of generations (default: 5000, reduced from 20K to limit exploitation)"
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        default=60,
        help="Parcel size in meters (must match GP training: 60, 120, or 240m)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/mapelites_gp/emit<N>_batch<M>_rep<R>)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU"
    )
    parser.add_argument(
        "--lambda-ucb",
        type=float,
        default=1.0,
        help="UCB penalty coefficient (score = mean - lambda * std). "
             "Penalizes uncertain predictions to prevent GP exploitation. "
             "SAIL uses 1.0. Set to 0 for pure mean (original behavior)."
    )
    parser.add_argument(
        "--genome-bounds",
        type=float,
        default=15.0,
        help="Clip genome values to [-bounds, bounds]. Training data has "
             "99th percentile at ±14. Set to 0 to disable clipping."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path("results") / "mapelites_gp" / f"emit{args.num_emitters}_batch{args.batch_size}_seed{args.seed}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Run experiment
    archive, history = run_mapelites(
        gp_model_path=Path(args.gp_model),
        num_emitters=args.num_emitters,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        parcel_size=args.parcel_size,
        output_dir=output_dir,
        seed=args.seed,
        device=device,
        lambda_ucb=args.lambda_ucb,
        genome_bounds=args.genome_bounds
    )
    
    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
