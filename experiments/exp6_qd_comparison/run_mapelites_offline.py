#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
MAP-Elites with Offline Surrogate Models (Experiment 6)

This script runs pure MAP-Elites optimization using pre-trained surrogate models:
1. U-Net baseline (pure exploitation)
2. SVGP baseline (pure exploitation)
3. SVGP + UCB (uncertainty-based exploration)
4. U-Net + SVGP hybrid (accuracy + exploration)

Checkpoints are saved every 1000 generations to enable recovery from failures.

Usage:
    # Config 1: U-Net baseline
    python experiments/exp6_qd_comparison/run_mapelites_offline.py \
        --model unet \
        --parcel-size 51 \
        --generations 10000 \
        --seed 42
    
    # Config 2: SVGP baseline
    python experiments/exp6_qd_comparison/run_mapelites_offline.py \
        --model svgp \
        --parcel-size 51 \
        --generations 10000 \
        --seed 42
    
    # Config 3: SVGP + UCB
    python experiments/exp6_qd_comparison/run_mapelites_offline.py \
        --model svgp \
        --ucb-lambda 1.0 \
        --parcel-size 51 \
        --generations 10000 \
        --seed 42
    
    # Config 4: U-Net + SVGP hybrid
    python experiments/exp6_qd_comparison/run_mapelites_offline.py \
        --model hybrid \
        --ucb-lambda 1.0 \
        --parcel-size 51 \
        --generations 10000 \
        --seed 42
"""

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp3_hpo.train_gp_hpo import SVGPModel
from models.unet import UNet, UNetConfig
from encodings.parametric import ParametricEncoding, NumbaFastEncoding, compute_features_batch_numba, numba_calculate_features
from domain_description.evaluation_klam import calculate_planning_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Optimized Domain Grid Construction (for U-Net)
# ============================================================================

def optimized_construct_domain_grids(
    heightmaps: np.ndarray,
    parcel_size_cells: int,
    grid_h: int = 66,
    grid_w: int = 94
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized domain grid construction matching KLAM_21 conventions.
    
    Reproduces the exact domain layout from evaluation_klam.compute_fitness_klam():
    - 2° terrain slope (elevation increases toward left/upwind edge)
    - Parcel position: centered vertically, shifted right by left_extension
      (left_extension = grid_w - grid_h = original_offset)
    - Landuse: 7 (free space) left of parcel, 2 (low-density buildings) from parcel onward
    
    Args:
        heightmaps: (N, D, D) array of heightmaps in floors
        parcel_size_cells: Parcel size in grid cells
        grid_h: Output grid height (rows, matches env_cells_y)
        grid_w: Output grid width (cols, matches env_cells_x = env_cells_y + left_extension)
    
    Returns:
        terrain: (N, grid_h, grid_w) elevation in meters (2° slope)
        buildings: (N, grid_h, grid_w) building heights in meters
        landuse: (N, grid_h, grid_w) landuse codes (7=free, 2=low-density)
    """
    batch_size = len(heightmaps)
    xy_scale = 3.0  # meters per cell (fixed project constant)
    
    # === Parcel placement matching KLAM convention ===
    # KLAM builds a square base grid (env_cells_base = grid_h), then extends left.
    # left_extension = original_offset = (grid_h - parcel_size_cells) // 2
    # So parcel x-offset = original_offset + left_extension = original_offset + (grid_w - grid_h)
    offset_h = (grid_h - parcel_size_cells) // 2
    offset_w = offset_h + (grid_w - grid_h)  # accounts for left extension
    
    # === Terrain: continuous 2° slope (matches KLAM) ===
    slope_gradient = np.tan(np.radians(2.0))  # ~0.0349 m/m
    col_indices = np.arange(grid_w, dtype=np.float32)
    distance_from_right_m = (grid_w - col_indices) * xy_scale
    elevation_profile = distance_from_right_m * slope_gradient  # (grid_w,)
    terrain = np.broadcast_to(
        elevation_profile[np.newaxis, np.newaxis, :],
        (batch_size, grid_h, grid_w)
    ).copy()  # copy() because broadcast_to returns read-only view
    
    # === Buildings ===
    buildings = np.zeros((batch_size, grid_h, grid_w), dtype=np.float32)
    buildings[:, offset_h:offset_h+parcel_size_cells, 
              offset_w:offset_w+parcel_size_cells] = heightmaps * 3.0  # floors → meters
    
    # === Landuse matching KLAM: free space left of parcel, buildings right ===
    landuse = np.ones((batch_size, grid_h, grid_w), dtype=np.float32)
    landuse[:, :, :offset_w] = 7   # Left side (upwind) = free space
    landuse[:, :, offset_w:] = 2   # From parcel left edge onward = low-density buildings
    
    return terrain, buildings, landuse


def compute_klam_roi_mask(
    grid_h: int, grid_w: int, parcel_size_cells: int
) -> np.ndarray:
    """
    Compute ROI mask matching KLAM_21's evaluation region.
    
    KLAM uses: parcel + 50% buffer in y-direction, parcel left edge - 50% buffer
    to right edge (full downwind extent) in x-direction.
    
    Parcel position follows KLAM convention:
      offset_h = (grid_h - parcel_size_cells) // 2
      offset_w = offset_h + (grid_w - grid_h)   [accounts for left extension]
    
    Args:
        grid_h: Grid height in cells
        grid_w: Grid width in cells  
        parcel_size_cells: Parcel size in cells
    
    Returns:
        roi_mask: (grid_h, grid_w) boolean mask
    """
    offset_h = (grid_h - parcel_size_cells) // 2
    offset_w = offset_h + (grid_w - grid_h)  # KLAM left extension
    buffer_size = int(parcel_size_cells * 0.5)
    
    roi_start_y = max(0, offset_h - buffer_size)
    roi_end_y = min(grid_h, offset_h + parcel_size_cells + buffer_size)
    roi_start_x = max(0, offset_w - buffer_size)
    roi_end_x = grid_w  # Extend to right edge (downwind)
    
    roi_mask = np.zeros((grid_h, grid_w), dtype=bool)
    roi_mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = True
    return roi_mask


# ============================================================================
# Model Loaders
# ============================================================================

class SVGPEvaluator:
    """Wrapper for SVGP model evaluation with optional UCB."""
    
    def __init__(self, model_path: Path, device: torch.device, ucb_lambda: float = 0.0):
        self.device = device
        self.ucb_lambda = ucb_lambda
        logger.info(f"Loading SVGP model from {model_path}")
        logger.info(f"  UCB lambda: {ucb_lambda}")
        
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
        
        logger.info("SVGP model loaded successfully")
    
    def predict(self, genomes: np.ndarray, parcel_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict objectives with SVGP.
        
        Args:
            genomes: (N, 60) array of genomes
            parcel_sizes: (N,) array of parcel sizes in meters
        
        Returns:
            objectives_mean: (N,) array of predicted objectives
            objectives_std: (N,) array of prediction uncertainties
        """
        # Prepare inputs (genome + 2x parcel_size for width/height)
        parcel_cols = np.column_stack([parcel_sizes, parcel_sizes])
        X = np.column_stack([genomes, parcel_cols]).astype(np.float32)
        
        # Convert to torch
        X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Normalize
        X_norm = (X_torch - self.train_x_mean) / (self.train_x_std + 1e-6)
        
        # Predict
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_norm))
            pred_mean = pred.mean * self.train_y_std + self.train_y_mean
            pred_std = pred.stddev * self.train_y_std
        
        objectives_mean = pred_mean.cpu().numpy()
        objectives_std = pred_std.cpu().numpy()
        
        return objectives_mean, objectives_std
    
    def evaluate(self, genomes: np.ndarray, parcel_sizes: np.ndarray, encoding_func=None) -> Dict:
        """
        Evaluate solutions with SVGP (with optional UCB).
        
        Args:
            genomes: (N, 60) array of genomes
            parcel_sizes: (N,) array of parcel sizes
            encoding_func: Unused, kept for interface consistency with U-Net
        
        Returns dictionary with objectives and metadata.
        """
        objectives_mean, objectives_std = self.predict(genomes, parcel_sizes)
        
        # Compute UCB-adjusted objectives
        objectives_adjusted = objectives_mean + self.ucb_lambda * objectives_std
        
        return {
            'objective_predicted': objectives_mean,
            'objective_adjusted': objectives_adjusted,
            'svgp_uncertainty': objectives_std,
        }


class UNetEvaluator:
    """Wrapper for U-Net model evaluation."""
    
    def __init__(self, model_path: Path, device: torch.device, parcel_size: int = 27, use_compile: bool = True, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16 and device.type == 'cuda'  # FP16 only on GPU
        logger.info(f"Loading U-Net model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Load config
        config = UNetConfig(**checkpoint['config'])
        logger.info(f"  Config: {config}")
        
        # Create model
        self.model = UNet(config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Apply FP16 optimization (10x speedup on A100)
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("  ✓ FP16 enabled")
        
        # Apply torch.compile optimization (4x speedup)
        if use_compile and hasattr(torch, 'compile'):
            logger.info("  Compiling model with torch.compile (this may take a minute)...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
            logger.info("  ✓ torch.compile enabled")
        
        # Load normalization parameters from separate JSON file
        norm_path = model_path.parent / 'normalization.json'
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
        
        with open(norm_path) as f:
            norm_stats = json.load(f)
        
        self.terrain_mean = norm_stats['input']['terrain']['mean']
        self.terrain_std = norm_stats['input']['terrain']['std']
        self.buildings_mean = norm_stats['input']['buildings']['mean']
        self.buildings_std = norm_stats['input']['buildings']['std']
        self.landuse_mean = norm_stats['input']['landuse']['mean']
        self.landuse_std = norm_stats['input']['landuse']['std']
        self.output_means = {k: v['mean'] for k, v in norm_stats['output'].items()}
        self.output_stds = {k: v['std'] for k, v in norm_stats['output'].items()}
        
        # Pre-initialize NumbaFastEncoding for reuse (avoids JIT recompilation)
        self.fast_encoding = NumbaFastEncoding(parcel_size=parcel_size)
        
        # Flag to track if we've done warmup
        self._warmup_done = False
        
        logger.info("U-Net model loaded successfully")
    
    def predict(self, terrain: np.ndarray, buildings: np.ndarray, landuse: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict KLAM fields from inputs.
        
        Args:
            terrain: (N, H, W) array
            buildings: (N, H, W) array
            landuse: (N, H, W) array
        
        Returns:
            Dictionary with predicted fields: Ex, Hx, uq, vq, uz, vz
        """
        # Normalize inputs
        terrain_norm = (terrain - self.terrain_mean) / self.terrain_std
        buildings_norm = (buildings - self.buildings_mean) / self.buildings_std
        landuse_norm = (landuse - self.landuse_mean) / self.landuse_std
        
        # Stack: (N, 3, H, W)
        X = np.stack([terrain_norm, buildings_norm, landuse_norm], axis=1)
        
        # Convert to tensor with appropriate dtype
        dtype = torch.float16 if self.use_fp16 else torch.float32
        X_torch = torch.tensor(X, dtype=dtype).to(self.device)
        
        # Predict
        with torch.no_grad():
            Y_pred = self.model(X_torch)
        
        # Convert back to float32 numpy
        Y_pred = Y_pred.float().cpu().numpy()
        
        # Denormalize outputs
        output_vars = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']
        predictions = {}
        for i, var in enumerate(output_vars):
            predictions[var] = Y_pred[:, i, :, :] * self.output_stds[var] + self.output_means[var]
        
        return predictions
    
    def compute_cold_air_flux(self, predictions: Dict[str, np.ndarray], roi_mask: np.ndarray) -> np.ndarray:
        """
        Compute cold air flux from U-Net predictions.
        
        Φ = mean(Ex) * mean(sqrt(uq^2 + vq^2))
        
        Args:
            predictions: Dictionary with Ex, uq, vq fields (N, H, W)
            roi_mask: (H, W) boolean mask for region of interest
        
        Returns:
            flux: (N,) array of cold air flux values
        """
        Ex = predictions['Ex']
        uq = predictions['uq']
        vq = predictions['vq']
        
        # Convert cm/s to m/s
        uq_ms = uq / 100.0
        vq_ms = vq / 100.0
        
        # Compute wind speed
        wind_speed = np.sqrt(uq_ms**2 + vq_ms**2)
        
        # Apply ROI mask and compute means
        flux = np.zeros(len(Ex))
        for i in range(len(Ex)):
            Ex_roi = Ex[i][roi_mask]
            wind_roi = wind_speed[i][roi_mask]
            flux[i] = np.mean(Ex_roi) * np.mean(wind_roi)
        
        return flux
    
    def evaluate(self, genomes: np.ndarray, parcel_sizes: np.ndarray, encoding_func) -> Dict:
        """
        Evaluate solutions with U-Net.
        
        NOTE: All parcel_sizes must match the U-Net training size.
        This is enforced in create_evaluator().
        
        Returns dictionary with objectives.
        """
        # Get grid dimensions from model config
        expected_h = self.model.config.input_height
        expected_w = self.model.config.input_width
        
        # Calculate parcel size in cells (assuming all parcels are same size)
        parcel_size_m = parcel_sizes[0]  # Should be uniform
        parcel_size_cells = int(parcel_size_m // 3)  # 3m resolution
        
        # Use pre-initialized NumbaFastEncoding (avoids JIT recompilation overhead)
        heightmaps = self.fast_encoding.express_batch(genomes)
        
        # Vectorized domain grid construction (~40× speedup)
        terrain, buildings, landuse = optimized_construct_domain_grids(
            heightmaps, 
            parcel_size_cells,
            grid_h=expected_h,
            grid_w=expected_w
        )
        
        # Predict with U-Net
        predictions = self.predict(terrain, buildings, landuse)
        
        # Compute cold air flux over KLAM-matching ROI (parcel + 50% buffer, extended downwind)
        roi_mask = compute_klam_roi_mask(expected_h, expected_w, parcel_size_cells)
        objectives = self.compute_cold_air_flux(predictions, roi_mask)
        
        # Compute features from heightmaps (avoid re-expressing genomes)
        pixel_size = self.fast_encoding.config['xy_scale']
        features = np.zeros((len(heightmaps), 8), dtype=np.float64)
        for i in range(len(heightmaps)):
            features[i] = numba_calculate_features(heightmaps[i], pixel_size)
        
        return {
            'objective_predicted': objectives,
            'objective_adjusted': objectives,  # No exploration bonus for U-Net alone
            'features': features,  # Return features to avoid double computation
        }


class HybridEvaluator:
    """Hybrid U-Net + SVGP evaluator (Config 4)."""
    
    def __init__(
        self,
        unet_path: Path,
        svgp_path: Path,
        device: torch.device,
        parcel_size: int = 27,
        ucb_lambda: float = 1.0,
        use_compile: bool = True,
        use_fp16: bool = True
    ):
        self.unet = UNetEvaluator(unet_path, device, parcel_size=parcel_size, use_compile=use_compile, use_fp16=use_fp16)
        self.svgp = SVGPEvaluator(svgp_path, device, ucb_lambda=0.0)  # No UCB in SVGP predict
        self.ucb_lambda = ucb_lambda
        
        logger.info(f"Hybrid evaluator: U-Net (accuracy) + SVGP (uncertainty)")
        logger.info(f"  UCB lambda: {ucb_lambda}")
    
    def evaluate(self, genomes: np.ndarray, parcel_sizes: np.ndarray, encoding_func) -> Dict:
        """
        Evaluate with hybrid: U-Net prediction + SVGP uncertainty.
        """
        # Get U-Net predictions (includes features)
        unet_result = self.unet.evaluate(genomes, parcel_sizes, encoding_func)
        unet_objectives = unet_result['objective_predicted']
        
        # Get SVGP uncertainty
        _, svgp_uncertainty = self.svgp.predict(genomes, parcel_sizes)
        
        # Compute UCB-adjusted objectives
        objectives_adjusted = unet_objectives + self.ucb_lambda * svgp_uncertainty
        
        return {
            'objective_predicted': unet_objectives,
            'objective_adjusted': objectives_adjusted,
            'svgp_uncertainty': svgp_uncertainty,
            'features': unet_result['features'],  # Pass through features
        }


# ============================================================================
# Feature Computation (uses optimized Numba implementation)
# ============================================================================

# Wrapper for backward compatibility - delegates to imported compute_features_batch_numba
def compute_features_batch(
    genomes: np.ndarray, 
    parcel_size: int, 
    encoding: Optional[NumbaFastEncoding] = None
) -> np.ndarray:
    """
    Compute planning features from genomes using Numba-optimized implementation.
    
    This is a compatibility wrapper around compute_features_batch_numba from
    encodings.parametric.fast_encoding.
    
    Args:
        genomes: (N, 60) array of genomes
        parcel_size: Fixed parcel size in meters (integer)
        encoding: Optional pre-initialized NumbaFastEncoding instance (for reuse)
    
    Returns:
        features: (N, 8) array of planning features
    """
    return compute_features_batch_numba(genomes, parcel_size, encoding)


# ============================================================================
# MAP-Elites Optimization
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    model_type: str  # 'unet', 'svgp', 'hybrid'
    ucb_lambda: float
    parcel_size: int
    num_generations: int
    num_emitters: int
    batch_size: int
    seed: int
    unet_model_path: Optional[Path]
    svgp_model_path: Optional[Path]
    output_dir: Path
    use_compile: bool = True   # torch.compile optimization (4x speedup)
    use_fp16: bool = True      # FP16 mixed precision (2x speedup, 10x with compile)


def create_evaluator(config: ExperimentConfig, device: torch.device):
    """Create appropriate evaluator based on config."""
    # For U-Net models, use size-specific model path if not provided
    unet_model_path = config.unet_model_path
    if config.model_type in ['unet', 'hybrid']:
        # If default path, auto-select based on parcel size
        if 'unet_experiment/sail_mse' in str(unet_model_path):
            unet_model_path = Path(f'results/exp5_unet/unet_experiment/unet_{config.parcel_size}m/best_model.pth')
            logger.info(f"Auto-selected U-Net model: {unet_model_path}")
        
        # Verify model exists
        if not unet_model_path.exists():
            raise FileNotFoundError(
                f"U-Net model not found: {unet_model_path}\n"
                f"Available parcel sizes: 60, 120, 240m\n"
                f"Train with: python experiments/exp5_unet/train_unet_klam.py --parcel-size {config.parcel_size}"
            )
    
    if config.model_type == 'unet':
        return UNetEvaluator(unet_model_path, device, parcel_size=config.parcel_size,
                            use_compile=config.use_compile, use_fp16=config.use_fp16)
    elif config.model_type == 'svgp':
        return SVGPEvaluator(config.svgp_model_path, device, config.ucb_lambda)
    elif config.model_type == 'hybrid':
        return HybridEvaluator(
            unet_model_path,
            config.svgp_model_path,
            device,
            parcel_size=config.parcel_size,
            ucb_lambda=config.ucb_lambda,
            use_compile=config.use_compile,
            use_fp16=config.use_fp16
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def run_mapelites(config: ExperimentConfig):
    """Run MAP-Elites optimization with offline surrogate."""
    
    logger.info("="*80)
    logger.info("MAP-ELITES WITH OFFLINE SURROGATE (EXPERIMENT 6)")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model type: {config.model_type}")
    logger.info(f"  UCB lambda: {config.ucb_lambda}")
    logger.info(f"  Parcel size: {config.parcel_size}m")
    logger.info(f"  Generations: {config.num_generations}")
    logger.info(f"  Emitters: {config.num_emitters}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Seed: {config.seed}")
    if config.model_type in ['unet', 'hybrid']:
        logger.info(f"  torch.compile: {config.use_compile}")
        logger.info(f"  FP16: {config.use_fp16}")
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create evaluator
    evaluator = create_evaluator(config, device)
    encoding_func = ParametricEncoding  # For compatibility
    
    # Pre-initialize NumbaFastEncoding for feature computation (reused across all generations)
    logger.info("Initializing NumbaFastEncoding for feature computation...")
    fast_encoding = NumbaFastEncoding(config.parcel_size)
    logger.info("  ✓ NumbaFastEncoding ready (~16× speedup vs original)")
    
    # Load domain config for features
    with open(project_root / 'domain_description' / 'cfg.yml') as f:
        domain_config = yaml.safe_load(f)
    
    feat_ranges_raw = domain_config['feat_ranges']
    feature_labels = domain_config['labels']
    
    # Convert feat_ranges from [mins, maxs] to list of (min, max) tuples
    feature_ranges = list(zip(feat_ranges_raw[0], feat_ranges_raw[1]))
    
    # Create archive
    logger.info("Creating GridArchive...")
    archive = GridArchive(
        solution_dim=60,  # Only evolve 60D genome, parcel size is fixed
        dims=[5] * 8,  # 8D feature space with 5 bins each
        ranges=feature_ranges,
        seed=config.seed,
        qd_score_offset=0.0,
    )
    
    # Create emitters
    logger.info(f"Creating {config.num_emitters} GaussianEmitters...")
    emitters = [
        GaussianEmitter(
            archive=archive,
            x0=np.zeros(60),  # Only evolve 60D genome
            sigma=0.5,
            batch_size=config.batch_size,
            seed=config.seed + i,
        )
        for i in range(config.num_emitters)
    ]
    
    # Create scheduler
    scheduler = Scheduler(archive, emitters)
    
    # Ensure output directory exists (needed for checkpoints during optimization)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimization loop
    logger.info("Starting optimization...")
    start_time = time.time()
    
    total_evaluations = 0
    log_interval = 1000
    
    for gen in range(config.num_generations):
        # Ask for solutions (60D genomes only)
        genomes = scheduler.ask()
        
        # Use fixed parcel size for all solutions
        parcel_sizes = np.full(len(genomes), config.parcel_size, dtype=np.float32)
        
        # Evaluate with surrogate
        if config.model_type == 'unet':
            result = evaluator.evaluate(genomes, parcel_sizes, encoding_func)
        else:  # svgp or hybrid
            result = evaluator.evaluate(genomes, parcel_sizes, encoding_func)
        
        # Extract objectives for archive
        # Use adjusted objectives for selection
        objectives_adjusted = result['objective_adjusted']
        
        # Compute features (use from evaluator if available to avoid double computation)
        if 'features' in result:
            features = result['features']
        else:
            # For SVGP/hybrid, compute features separately
            features = compute_features_batch(genomes, config.parcel_size, encoding=fast_encoding)
        
        # Tell scheduler (use adjusted objectives)
        scheduler.tell(objectives_adjusted, features)
        
        total_evaluations += len(genomes)
        
        # Save checkpoint every 1000 generations
        if (gen + 1) % 1000 == 0:
            checkpoint_path = config.output_dir / f"checkpoint_gen{gen+1}.pkl"
            logger.info(f"Saving checkpoint at generation {gen+1}...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'archive': archive,
                    'generation': gen + 1,
                    'total_evaluations': total_evaluations,
                    'config': vars(config),
                    'elapsed_time': time.time() - start_time,
                }, f)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Logging
        if (gen + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            qd_score = archive.stats.qd_score
            coverage = archive.stats.coverage
            
            logger.info(
                f"Gen {gen+1}/{config.num_generations} | "
                f"QD={qd_score:.2f} | "
                f"Cov={coverage:.2%} | "
                f"Time={elapsed:.1f}s"
            )
    
    # Final statistics
    elapsed = time.time() - start_time
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Total evaluations: {total_evaluations}")
    logger.info(f"QD score: {archive.stats.qd_score:.2f}")
    logger.info(f"Coverage: {archive.stats.coverage:.2%}")
    logger.info(f"Archive size: {len(archive)}")
    
    # Save archive
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment name
    exp_name = f"{config.model_type}"
    if config.ucb_lambda > 0:
        exp_name += f"_ucb{config.ucb_lambda}"
    exp_name += f"_size{config.parcel_size}_seed{config.seed}"
    
    archive_path = config.output_dir / f"archive_{exp_name}.pkl"
    logger.info(f"Saving archive to {archive_path}")
    
    with open(archive_path, 'wb') as f:
        pickle.dump(archive, f)
    
    # Save metadata
    metadata = {
        'config': {
            'model_type': config.model_type,
            'ucb_lambda': config.ucb_lambda,
            'parcel_size': config.parcel_size,
            'num_generations': config.num_generations,
            'num_emitters': config.num_emitters,
            'batch_size': config.batch_size,
            'seed': config.seed,
        },
        'results': {
            'qd_score': float(archive.stats.qd_score),
            'coverage': float(archive.stats.coverage),
            'archive_size': len(archive),
            'total_evaluations': total_evaluations,
            'elapsed_time': elapsed,
        },
        'feature_ranges': feature_ranges,
        'feature_labels': feature_labels,
    }
    
    metadata_path = config.output_dir / f"metadata_{exp_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Done!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MAP-Elites with Offline Surrogates")
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'svgp', 'hybrid'],
                       help='Model type: unet, svgp, or hybrid')
    parser.add_argument('--ucb-lambda', type=float, default=0.0,
                       help='UCB exploration parameter (0 = pure exploitation)')
    
    # Model paths
    parser.add_argument('--unet-model', type=str,
                       default='results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth',
                       help='Path to U-Net model checkpoint')
    parser.add_argument('--svgp-model', type=str,
                       default='results/exp3_hpo/hyperparameterization/model_optimized_ind2500_kmeans_rep1.pth',
                       help='Path to SVGP model checkpoint')
    
    # Optimization parameters
    parser.add_argument('--parcel-size', type=int, default=120,
                       choices=[60, 120, 240],
                       help='Parcel size in meters (120m recommended)')
    parser.add_argument('--generations', type=int, default=10000,
                       help='Number of generations')
    parser.add_argument('--num-emitters', type=int, default=128,
                       help='Number of emitters')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per emitter')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                       default='results/exp6_qd_comparison',
                       help='Output directory')
    
    # Performance optimization (U-Net only)
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile optimization (default: enabled)')
    parser.add_argument('--no-fp16', action='store_true',
                       help='Disable FP16 mixed precision (default: enabled)')
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        model_type=args.model,
        ucb_lambda=args.ucb_lambda,
        parcel_size=args.parcel_size,
        num_generations=args.generations,
        num_emitters=args.num_emitters,
        batch_size=args.batch_size,
        seed=args.seed,
        unet_model_path=Path(args.unet_model) if args.model in ['unet', 'hybrid'] else None,
        svgp_model_path=Path(args.svgp_model) if args.model in ['svgp', 'hybrid'] else None,
        output_dir=Path(args.output_dir),
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16,
    )
    
    # Run experiment
    run_mapelites(config)


if __name__ == '__main__':
    main()
