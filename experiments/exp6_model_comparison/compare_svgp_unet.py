#!/usr/bin/env python
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
WP1.2: SVGP vs Small U-Net Comparison

Compares offline SVGP surrogate model with small U-Net to determine
which model to use for MAP-Elites optimization in the GUI.

Decision criteria:
- Throughput (samples/sec)
- Agreement (Spearman ρ correlation)
- Ranking fidelity (Top-100 overlap)
- Calibration (fitness distribution match)

Usage:
    python experiments/exp6_model_comparison/compare_svgp_unet.py \
        --svgp-model results/hyperparameterization/model_optimized_ind2500_random_rep1.pth \
        --unet-model results/unet_experiment/sail_mse_seed42/best_model.pth \
        --num-samples 1000 \
        --parcel-size 51
"""

import sys
import logging
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from scipy.stats import qmc, spearmanr, pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unet import UNet, UNetConfig
from encodings.parametric import ParametricEncoding  # Uses NumbaFastEncoding (16× faster)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SVGP Model Definition
# ============================================================================

class SVGPModel(ApproximateGP):
    """SVGP model for cold air flux prediction."""
    
    def __init__(self, inducing_points, input_dim=62):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ============================================================================
# Model Loading
# ============================================================================

def load_svgp_model(model_path: Path, device: torch.device) -> Tuple:
    """Load SVGP model and likelihood."""
    logger.info(f"Loading SVGP model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get inducing points from state dict
    inducing_key = 'variational_strategy.inducing_points'
    inducing_points = checkpoint['model_state_dict'][inducing_key]
    num_inducing = inducing_points.size(0)
    input_dim = inducing_points.size(1)
    
    logger.info(f"  Inducing points: {num_inducing}")
    logger.info(f"  Input dimension: {input_dim}")
    
    # Create model and likelihood
    model = SVGPModel(inducing_points.to(device), input_dim=input_dim).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # Load normalization parameters
    model.train_x_mean = checkpoint['train_x_mean'].to(device)
    model.train_x_std = checkpoint['train_x_std'].to(device)
    model.train_y_mean = checkpoint['train_y_mean'].to(device)
    model.train_y_std = checkpoint['train_y_std'].to(device)
    
    model.eval()
    likelihood.eval()
    
    return model, likelihood


def load_unet_model(model_dir: Path, device: torch.device) -> Tuple:
    """Load U-Net model and normalization stats.
    
    Note: U-Net is loaded to CPU to avoid GPU OOM issues with limited VRAM.
    """
    logger.info(f"Loading U-Net model from {model_dir}")
    
    # Load config (may be standalone config.json or nested in results.json)
    config_path = model_dir / 'config.json'
    results_path = model_dir / 'results.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        config_dict = results['config']
    else:
        raise FileNotFoundError(f"No config.json or results.json found in {model_dir}")
    
    config = UNetConfig(**config_dict)
    
    # Load model to CPU (U-Net is too large for 2GB GPU)
    cpu_device = torch.device('cpu')
    model = UNet(config).to(cpu_device)
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=cpu_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load normalization stats
    norm_path = model_dir / 'normalization.json'
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    
    logger.info(f"  Input size: {config.input_height}×{config.input_width}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Device: CPU (to avoid GPU OOM)")
    
    return model, norm_stats, config


# ============================================================================
# Data Generation
# ============================================================================

def generate_diverse_layouts(
    num_samples: int,
    parcel_size: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate diverse building layouts using Sobol sequence.
    
    Returns:
        genomes: (N, 60) array
        widths: (N,) array
        heights: (N,) array
    """
    logger.info(f"Generating {num_samples} diverse layouts (parcel={parcel_size}m)")
    
    # Initialize encoding
    encoding = ParametricEncoding()
    genome_length = encoding.get_dimension()
    
    # Generate Sobol sequence
    sampler = qmc.Sobol(d=genome_length, scramble=True, seed=seed)
    genomes = sampler.random(num_samples)
    
    # Scale from [0, 1] to [-1, 1]
    genomes = 2 * genomes - 1
    
    # Parcel dimensions
    widths = np.full(num_samples, parcel_size, dtype=np.float32)
    heights = np.full(num_samples, parcel_size, dtype=np.float32)
    
    return genomes, widths, heights


# ============================================================================
# SVGP Evaluation
# ============================================================================

def evaluate_with_svgp(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    genomes: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    batch_size: int = 256,
    device: torch.device = None
) -> Tuple[np.ndarray, float]:
    """
    Evaluate layouts with SVGP model.
    
    Returns:
        predictions: (N,) fitness values
        throughput: samples/second
    """
    logger.info("Evaluating with SVGP...")
    
    # Prepare input: [genomes (60D), width, height] = 62D
    X = np.column_stack([genomes, widths, heights])
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Normalize
    X_normalized = (X_tensor - model.train_x_mean) / model.train_x_std
    
    # Predict in batches
    predictions = []
    start_time = time.time()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X_normalized), batch_size):
            batch = X_normalized[i:i+batch_size]
            pred_dist = likelihood(model(batch))
            pred_mean = pred_dist.mean
            
            # Denormalize
            pred_mean = pred_mean * model.train_y_std + model.train_y_mean
            predictions.append(pred_mean.cpu().numpy())
    
    elapsed = time.time() - start_time
    predictions = np.concatenate(predictions)
    throughput = len(predictions) / elapsed
    
    logger.info(f"  Throughput: {throughput:.1f} samples/s")
    logger.info(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions, throughput


# ============================================================================
# U-Net Evaluation
# ============================================================================

def create_unet_inputs(
    genomes: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    norm_stats: Dict,
    config: UNetConfig,
    encoding: ParametricEncoding
) -> torch.Tensor:
    """
    Create normalized U-Net inputs from genomes.
    
    Returns:
        inputs: (N, 3, H, W) tensor
    """
    num_samples = len(genomes)
    inputs = np.zeros((num_samples, 3, config.input_height, config.input_width), dtype=np.float32)
    
    for i in range(num_samples):
        # Express genome to heightmap (floors)
        # Note: express() only takes genome and as_height_map parameters
        phenotype_floors = encoding.express(genomes[i], as_height_map=True)
        
        # Convert floors to meters (3m per floor)
        buildings = phenotype_floors * 3.0
        
        # Create terrain (flat for generic model)
        terrain = np.zeros_like(buildings)
        
        # Create landuse (simple: 2=buildings, 7=free space)
        landuse = np.where(buildings > 0, 2, 7).astype(np.float32)
        
        # Pad/resize to match U-Net input size
        # The phenotype size depends on encoding config (e.g., 20×20)
        # U-Net expects (66, 94)
        h_pheno, w_pheno = terrain.shape
        h_pad = max(0, config.input_height - h_pheno)
        w_pad = max(0, config.input_width - w_pheno)
        
        # Pad symmetrically (center the design)
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        
        terrain_padded = np.pad(terrain, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        buildings_padded = np.pad(buildings, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        landuse_padded = np.pad(landuse, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=7)  # 7=free space
        
        # Crop if needed (shouldn't happen with current sizes)
        terrain_padded = terrain_padded[:config.input_height, :config.input_width]
        buildings_padded = buildings_padded[:config.input_height, :config.input_width]
        landuse_padded = landuse_padded[:config.input_height, :config.input_width]
        
        inputs[i, 0] = terrain_padded
        inputs[i, 1] = buildings_padded
        inputs[i, 2] = landuse_padded
    
    # Normalize
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    for c, var_name in enumerate(['terrain', 'buildings', 'landuse']):
        mean = norm_stats['input'][var_name]['mean']
        std = norm_stats['input'][var_name]['std']
        inputs_tensor[:, c] = (inputs_tensor[:, c] - mean) / std
    
    return inputs_tensor


def compute_fitness_from_unet_output(
    outputs: torch.Tensor,
    norm_stats: Dict
) -> np.ndarray:
    """
    Compute cold air flux from U-Net outputs.
    
    Matches evaluation_klam.py formula:
    Fitness = mean(Ex) × mean(wind_speed_2m)
    
    Units:
    - Ex: 100 J/m² (KLAM raw output, kept as-is)
    - uq, vq: cm/s (KLAM raw output) → converted to m/s
    - Result: (100 J/m²) × (m/s) = 100 W/m²
    """
    # Denormalize outputs
    # Output order: Ex, Hx, uq, vq, uz, vz (channels 0-5)
    outputs_denorm = torch.zeros_like(outputs)
    
    output_vars = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']
    for c, var_name in enumerate(output_vars):
        mean = norm_stats['output'][var_name]['mean']
        std = norm_stats['output'][var_name]['std']
        outputs_denorm[:, c] = outputs[:, c] * std + mean
    
    # Extract fields (in KLAM raw units)
    Ex = outputs_denorm[:, 0]  # Cold air content [100 J/m²]
    uq = outputs_denorm[:, 2]  # u-velocity at 2m [cm/s]
    vq = outputs_denorm[:, 3]  # v-velocity at 2m [cm/s]
    
    # Convert velocities from cm/s to m/s (matching evaluation_klam.py)
    uq_ms = uq / 100.0
    vq_ms = vq / 100.0
    
    # Compute wind speed at 2m height [m/s]
    wind_speed_2m = torch.sqrt(uq_ms**2 + vq_ms**2)
    
    # Cold Air Flux: Φ = Ex × u_2m
    # Units: (100 J/m²) × (m/s) = 100 W/m²
    # This matches evaluation_klam.py line 411
    fitness = Ex.mean(dim=(1, 2)) * wind_speed_2m.mean(dim=(1, 2))
    
    return fitness.cpu().numpy()


def evaluate_with_unet(
    model: UNet,
    norm_stats: Dict,
    config: UNetConfig,
    genomes: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    batch_size: int = 8,
    device: torch.device = None  # Ignored - U-Net runs on CPU
) -> Tuple[np.ndarray, float]:
    """
    Evaluate layouts with U-Net model.
    
    Note: U-Net runs on CPU to avoid GPU OOM issues.
    
    Returns:
        predictions: (N,) fitness values
        throughput: samples/second
    """
    logger.info("Evaluating with U-Net...")
    
    # U-Net runs on CPU
    cpu_device = torch.device('cpu')
    
    # Create encoding
    encoding = ParametricEncoding()
    
    # Create inputs
    inputs = create_unet_inputs(genomes, widths, heights, norm_stats, config, encoding)
    inputs = inputs.to(cpu_device)  # Ensure CPU
    
    # Predict in batches
    predictions = []
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            outputs = model(batch)
            fitness = compute_fitness_from_unet_output(outputs, norm_stats)
            predictions.append(fitness)
    
    elapsed = time.time() - start_time
    predictions = np.concatenate(predictions)
    throughput = len(predictions) / elapsed
    
    logger.info(f"  Throughput: {throughput:.1f} samples/s")
    logger.info(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions, throughput


# ============================================================================
# Comparison Metrics
# ============================================================================

def compute_agreement_metrics(
    svgp_preds: np.ndarray,
    unet_preds: np.ndarray
) -> Dict:
    """Compute agreement metrics between models."""
    
    # Correlation
    pearson_r, pearson_p = pearsonr(svgp_preds, unet_preds)
    spearman_r, spearman_p = spearmanr(svgp_preds, unet_preds)
    
    # Ranking fidelity: Top-100 overlap
    top_k = min(100, len(svgp_preds))
    svgp_top_idx = np.argsort(svgp_preds)[-top_k:]
    unet_top_idx = np.argsort(unet_preds)[-top_k:]
    overlap = len(np.intersect1d(svgp_top_idx, unet_top_idx))
    overlap_pct = 100.0 * overlap / top_k
    
    # Normalized RMSE
    rmse = np.sqrt(np.mean((svgp_preds - unet_preds)**2))
    rmse_normalized = rmse / np.std(svgp_preds)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'top100_overlap': int(overlap),
        'top100_overlap_pct': float(overlap_pct),
        'rmse': float(rmse),
        'rmse_normalized': float(rmse_normalized)
    }


def make_recommendation(
    metrics: Dict,
    svgp_throughput: float,
    unet_throughput: float
) -> Dict:
    """
    Make recommendation based on metrics.
    
    Decision criteria from FINAL_WEEK_IMPLEMENTATION_PLAN:
    - If ρ > 0.85 AND U-Net throughput > 10× SVGP → Use U-Net
    - If ρ > 0.90 AND similar throughput → Use SVGP (more mature)
    - Else → Use both (ensemble?)
    """
    spearman_r = metrics['spearman_r']
    throughput_ratio = unet_throughput / svgp_throughput
    
    if spearman_r > 0.85 and throughput_ratio > 10:
        recommendation = "Use U-Net"
        reason = f"High agreement (ρ={spearman_r:.3f}) + U-Net is {throughput_ratio:.1f}× faster"
    elif spearman_r > 0.90:
        recommendation = "Use SVGP"
        reason = f"Very high agreement (ρ={spearman_r:.3f}), SVGP is more mature"
    elif spearman_r > 0.80:
        recommendation = "Either model works"
        reason = f"Good agreement (ρ={spearman_r:.3f}), choose based on integration ease"
    else:
        recommendation = "Use both (ensemble)"
        reason = f"Moderate agreement (ρ={spearman_r:.3f}), models complement each other"
    
    return {
        'recommendation': recommendation,
        'reason': reason,
        'spearman_r': spearman_r,
        'throughput_ratio': throughput_ratio
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(
    svgp_preds: np.ndarray,
    unet_preds: np.ndarray,
    metrics: Dict,
    output_path: Path
):
    """Create comparison plots."""
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(svgp_preds, unet_preds, alpha=0.3, s=10)
    ax1.plot([svgp_preds.min(), svgp_preds.max()], 
             [svgp_preds.min(), svgp_preds.max()], 
             'r--', label='Perfect agreement')
    ax1.set_xlabel('SVGP Prediction')
    ax1.set_ylabel('U-Net Prediction')
    ax1.set_title(f'Model Agreement\nSpearman ρ = {metrics["spearman_r"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = unet_preds - svgp_preds
    ax2.scatter(svgp_preds, residuals, alpha=0.3, s=10)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('SVGP Prediction')
    ax2.set_ylabel('Residual (U-Net - SVGP)')
    ax2.set_title(f'Residuals\nRMSE = {metrics["rmse"]:.2f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distributions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(svgp_preds, bins=50, alpha=0.5, label='SVGP', density=True)
    ax3.hist(unet_preds, bins=50, alpha=0.5, label='U-Net', density=True)
    ax3.set_xlabel('Fitness')
    ax3.set_ylabel('Density')
    ax3.set_title('Fitness Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Ranking comparison (top solutions)
    ax4 = fig.add_subplot(gs[1, 0])
    top_k = min(100, len(svgp_preds))
    svgp_ranks = np.argsort(np.argsort(svgp_preds))
    unet_ranks = np.argsort(np.argsort(unet_preds))
    svgp_top_idx = np.argsort(svgp_preds)[-top_k:]
    ax4.scatter(svgp_ranks[svgp_top_idx], unet_ranks[svgp_top_idx], alpha=0.5, s=20)
    ax4.plot([0, len(svgp_preds)], [0, len(svgp_preds)], 'r--', alpha=0.5)
    ax4.set_xlabel('SVGP Rank')
    ax4.set_ylabel('U-Net Rank')
    ax4.set_title(f'Top-{top_k} Ranking\nOverlap = {metrics["top100_overlap_pct"]:.1f}%')
    ax4.grid(True, alpha=0.3)
    
    # 5. Bland-Altman plot
    ax5 = fig.add_subplot(gs[1, 1])
    mean_pred = (svgp_preds + unet_preds) / 2
    diff = unet_preds - svgp_preds
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    ax5.scatter(mean_pred, diff, alpha=0.3, s=10)
    ax5.axhline(mean_diff, color='r', linestyle='-', label=f'Mean = {mean_diff:.2f}')
    ax5.axhline(mean_diff + 1.96*std_diff, color='r', linestyle='--', label=f'±1.96 SD')
    ax5.axhline(mean_diff - 1.96*std_diff, color='r', linestyle='--')
    ax5.set_xlabel('Mean Prediction')
    ax5.set_ylabel('Difference (U-Net - SVGP)')
    ax5.set_title('Bland-Altman Plot')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"""
    AGREEMENT METRICS
    ─────────────────
    Pearson r:     {metrics['pearson_r']:.3f}
    Spearman ρ:    {metrics['spearman_r']:.3f}
    
    RANKING FIDELITY
    ─────────────────
    Top-100 overlap: {metrics['top100_overlap_pct']:.1f}%
    
    ERROR METRICS
    ─────────────────
    RMSE:          {metrics['rmse']:.2f}
    Normalized RMSE: {metrics['rmse_normalized']:.3f}
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='WP1.2: Compare SVGP vs U-Net')
    
    parser.add_argument('--svgp-model', type=str, required=True,
                        help='Path to SVGP model checkpoint (.pth)')
    parser.add_argument('--unet-model', type=str, required=True,
                        help='Path to U-Net model directory')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of diverse layouts to evaluate')
    parser.add_argument('--parcel-size', type=int, default=51,
                        help='Parcel size in meters')
    parser.add_argument('--svgp-batch-size', type=int, default=256,
                        help='Batch size for SVGP evaluation')
    parser.add_argument('--unet-batch-size', type=int, default=8,
                        help='Batch size for U-Net evaluation (reduce if GPU OOM)')
    parser.add_argument('--output-dir', type=str, 
                        default='results/model_comparison',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if not args.no_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("\n" + "="*70)
    logger.info("LOADING MODELS")
    logger.info("="*70)
    
    svgp_model, svgp_likelihood = load_svgp_model(Path(args.svgp_model), device)
    unet_model, norm_stats, unet_config = load_unet_model(Path(args.unet_model), device)
    
    # Generate diverse layouts
    logger.info("\n" + "="*70)
    logger.info("GENERATING TEST DATA")
    logger.info("="*70)
    
    genomes, widths, heights = generate_diverse_layouts(
        args.num_samples, args.parcel_size, args.seed
    )
    
    # Evaluate with SVGP
    logger.info("\n" + "="*70)
    logger.info("SVGP EVALUATION")
    logger.info("="*70)
    
    svgp_preds, svgp_throughput = evaluate_with_svgp(
        svgp_model, svgp_likelihood, genomes, widths, heights,
        batch_size=args.svgp_batch_size, device=device
    )
    
    # Clear GPU memory before U-Net (SVGP takes significant memory)
    del svgp_model
    del svgp_likelihood
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("Cleared SVGP model from GPU memory")
    
    # Evaluate with U-Net
    logger.info("\n" + "="*70)
    logger.info("U-NET EVALUATION")
    logger.info("="*70)
    
    unet_preds, unet_throughput = evaluate_with_unet(
        unet_model, norm_stats, unet_config, genomes, widths, heights,
        batch_size=args.unet_batch_size, device=device
    )
    
    # Compute metrics
    logger.info("\n" + "="*70)
    logger.info("COMPARISON METRICS")
    logger.info("="*70)
    
    metrics = compute_agreement_metrics(svgp_preds, unet_preds)
    
    logger.info(f"\nAgreement:")
    logger.info(f"  Pearson r:  {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.2e})")
    logger.info(f"  Spearman ρ: {metrics['spearman_r']:.3f} (p={metrics['spearman_p']:.2e})")
    
    logger.info(f"\nRanking Fidelity:")
    logger.info(f"  Top-100 overlap: {metrics['top100_overlap']}/{100} ({metrics['top100_overlap_pct']:.1f}%)")
    
    logger.info(f"\nError:")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  Normalized RMSE: {metrics['rmse_normalized']:.3f}")
    
    logger.info(f"\nThroughput:")
    logger.info(f"  SVGP:  {svgp_throughput:.1f} samples/s")
    logger.info(f"  U-Net: {unet_throughput:.1f} samples/s")
    logger.info(f"  Ratio: {unet_throughput/svgp_throughput:.1f}×")
    
    # Make recommendation
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    
    recommendation = make_recommendation(metrics, svgp_throughput, unet_throughput)
    
    logger.info(f"\n✓ {recommendation['recommendation']}")
    logger.info(f"  Reason: {recommendation['reason']}")
    
    # Save results
    results = {
        'config': {
            'num_samples': args.num_samples,
            'parcel_size': args.parcel_size,
            'svgp_model': str(args.svgp_model),
            'unet_model': str(args.unet_model),
            'seed': args.seed
        },
        'throughput': {
            'svgp': svgp_throughput,
            'unet': unet_throughput,
            'ratio': unet_throughput / svgp_throughput
        },
        'metrics': metrics,
        'recommendation': recommendation,
        'predictions': {
            'svgp_mean': float(svgp_preds.mean()),
            'svgp_std': float(svgp_preds.std()),
            'svgp_min': float(svgp_preds.min()),
            'svgp_max': float(svgp_preds.max()),
            'unet_mean': float(unet_preds.mean()),
            'unet_std': float(unet_preds.std()),
            'unet_min': float(unet_preds.min()),
            'unet_max': float(unet_preds.max()),
        }
    }
    
    # Save JSON
    output_json = output_dir / 'svgp_vs_unet_comparison.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {output_json}")
    
    # Save predictions
    output_npz = output_dir / 'predictions.npz'
    np.savez(
        output_npz,
        genomes=genomes,
        widths=widths,
        heights=heights,
        svgp_predictions=svgp_preds,
        unet_predictions=unet_preds
    )
    logger.info(f"Saved predictions to {output_npz}")
    
    # Create plots
    plot_path = output_dir / 'comparison_plots.png'
    plot_comparison(svgp_preds, unet_preds, metrics, plot_path)
    
    logger.info("\n" + "="*70)
    logger.info("WP1.2 COMPLETE")
    logger.info("="*70)
    logger.info(f"\nNext step: Review results in {output_dir}")
    logger.info(f"Then proceed with recommended model for GUI integration")


if __name__ == '__main__':
    main()
