#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Train U-NET model for KLAM_21 spatial prediction.

This script trains a U-NET to predict KLAM_21 output fields (Ex, Hx, uq, vq, uz, vz)
from terrain, buildings, and landuse inputs.

Usage:
    # Smoke test (quick run)
    python experiments/train_unet_klam.py --data-type sail --loss mse --epochs 10 --smoke-test
    
    # Full training
    python experiments/train_unet_klam.py --data-type sail --loss mse --epochs 200 --seed 42
    python experiments/train_unet_klam.py --data-type random --loss mse_grad --epochs 200 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from models.unet import UNet, UNetConfig, get_loss_function

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class KLAMSpatialDataset(Dataset):
    """
    Dataset for KLAM_21 spatial prediction.
    
    Loads terrain, buildings, landuse as inputs and
    Ex, Hx, uq, vq, uz, vz at final timestamp as outputs.
    """
    
    # Output variable names in order
    OUTPUT_VARS = ['Ex', 'Hx', 'uq', 'vq', 'uz', 'vz']
    
    def __init__(
        self,
        data_files: List[Path],
        timestamp_idx: int = -1,  # -1 = last timestamp (14400s)
        normalize: bool = True,
    ):
        """
        Args:
            data_files: List of spatial .npz files to load
            timestamp_idx: Which timestamp to use (-1 = last = 14400s)
            normalize: Whether to normalize inputs and outputs
        """
        self.timestamp_idx = timestamp_idx
        self.normalize = normalize
        
        # Load and concatenate all data files
        logger.info(f"Loading {len(data_files)} data files...")
        
        all_terrain = []
        all_buildings = []
        all_landuse = []
        all_outputs = {var: [] for var in self.OUTPUT_VARS}
        
        for fpath in data_files:
            logger.info(f"  Loading {fpath.name}...")
            data = np.load(fpath)
            
            # Inputs
            all_terrain.append(data['terrain'].astype(np.float32))
            all_buildings.append(data['buildings'].astype(np.float32))
            all_landuse.append(data['landuse'].astype(np.float32))
            
            # Outputs (select timestamp)
            for var in self.OUTPUT_VARS:
                # Shape: (N, T, H, W) -> (N, H, W)
                arr = data[var][:, timestamp_idx, :, :].astype(np.float32)
                all_outputs[var].append(arr)
        
        # Concatenate
        self.terrain = np.concatenate(all_terrain, axis=0)
        self.buildings = np.concatenate(all_buildings, axis=0)
        self.landuse = np.concatenate(all_landuse, axis=0)
        self.outputs = {var: np.concatenate(all_outputs[var], axis=0) 
                       for var in self.OUTPUT_VARS}
        
        logger.info(f"  Total samples: {len(self.terrain)}")
        logger.info(f"  Grid shape: {self.terrain.shape[1:]} (H×W)")
        
        # Deduplicate based on building layouts
        self._deduplicate()
        
        # Compute normalization statistics
        if normalize:
            self._compute_normalization()
    
    def _deduplicate(self):
        """Remove duplicate samples based on building heightmaps."""
        logger.info("Deduplicating samples...")
        
        n_original = len(self.buildings)
        
        # Hash based on flattened buildings array
        # Round to 2 decimals to handle floating point noise
        buildings_rounded = np.round(self.buildings, decimals=2)
        
        # Create unique hash for each sample
        unique_hashes = set()
        unique_indices = []
        
        for i in range(len(buildings_rounded)):
            h = hash(buildings_rounded[i].tobytes())
            if h not in unique_hashes:
                unique_hashes.add(h)
                unique_indices.append(i)
        
        unique_indices = np.array(unique_indices)
        
        # Filter all arrays
        self.terrain = self.terrain[unique_indices]
        self.buildings = self.buildings[unique_indices]
        self.landuse = self.landuse[unique_indices]
        self.outputs = {var: arr[unique_indices] for var, arr in self.outputs.items()}
        
        n_unique = len(unique_indices)
        n_duplicates = n_original - n_unique
        
        logger.info(f"  Removed {n_duplicates} duplicates ({n_duplicates/n_original*100:.1f}%)")
        logger.info(f"  Unique samples: {n_unique}")
    
    def _compute_normalization(self):
        """Compute mean and std for normalization."""
        logger.info("Computing normalization statistics...")
        
        # Input statistics
        self.terrain_mean = np.mean(self.terrain)
        self.terrain_std = np.std(self.terrain) + 1e-6
        self.buildings_mean = np.mean(self.buildings)
        self.buildings_std = np.std(self.buildings) + 1e-6
        self.landuse_mean = np.mean(self.landuse)
        self.landuse_std = np.std(self.landuse) + 1e-6
        
        # Output statistics
        self.output_means = {var: np.mean(arr) for var, arr in self.outputs.items()}
        self.output_stds = {var: np.std(arr) + 1e-6 for var, arr in self.outputs.items()}
        
        logger.info("  Input stats:")
        logger.info(f"    terrain:   μ={self.terrain_mean:.2f}, σ={self.terrain_std:.2f}")
        logger.info(f"    buildings: μ={self.buildings_mean:.2f}, σ={self.buildings_std:.2f}")
        logger.info(f"    landuse:   μ={self.landuse_mean:.2f}, σ={self.landuse_std:.2f}")
        logger.info("  Output stats:")
        for var in self.OUTPUT_VARS:
            logger.info(f"    {var}: μ={self.output_means[var]:.2f}, σ={self.output_stds[var]:.2f}")
    
    def __len__(self) -> int:
        return len(self.terrain)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get raw data
        terrain = self.terrain[idx]
        buildings = self.buildings[idx]
        landuse = self.landuse[idx]
        
        # Normalize inputs
        if self.normalize:
            terrain = (terrain - self.terrain_mean) / self.terrain_std
            buildings = (buildings - self.buildings_mean) / self.buildings_std
            landuse = (landuse - self.landuse_mean) / self.landuse_std
        
        # Stack inputs: (3, H, W)
        x = np.stack([terrain, buildings, landuse], axis=0)
        
        # Stack outputs: (6, H, W)
        outputs = []
        for var in self.OUTPUT_VARS:
            out = self.outputs[var][idx]
            if self.normalize:
                out = (out - self.output_means[var]) / self.output_stds[var]
            outputs.append(out)
        y = np.stack(outputs, axis=0)
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def get_normalization_params(self) -> Dict:
        """Return normalization parameters for inference."""
        return {
            'input': {
                'terrain': {'mean': self.terrain_mean, 'std': self.terrain_std},
                'buildings': {'mean': self.buildings_mean, 'std': self.buildings_std},
                'landuse': {'mean': self.landuse_mean, 'std': self.landuse_std},
            },
            'output': {
                var: {'mean': self.output_means[var], 'std': self.output_stds[var]}
                for var in self.OUTPUT_VARS
            }
        }


# ============================================================================
# Training
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_x)
    
    return total_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            total_loss += loss.item() * len(batch_x)
    
    return total_loss / len(dataloader.dataset)


def compute_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_vars: List[str],
) -> Dict:
    """Compute evaluation metrics per output variable."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            target = batch_y.numpy()
            
            all_preds.append(pred)
            all_targets.append(target)
    
    preds = np.concatenate(all_preds, axis=0)  # (N, 6, H, W)
    targets = np.concatenate(all_targets, axis=0)
    
    metrics = {}
    
    for i, var in enumerate(output_vars):
        pred_var = preds[:, i]  # (N, H, W)
        target_var = targets[:, i]
        
        # Flatten for metrics
        pred_flat = pred_var.flatten()
        target_flat = target_var.flatten()
        
        # MSE
        mse = np.mean((pred_flat - target_flat) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        # R²
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        metrics[var] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
        }
    
    # Overall metrics
    metrics['overall'] = {
        'mse': float(np.mean([m['mse'] for m in metrics.values() if isinstance(m, dict)])),
        'mae': float(np.mean([m['mae'] for m in metrics.values() if isinstance(m, dict)])),
        'r2': float(np.mean([m['r2'] for m in metrics.values() if isinstance(m, dict)])),
    }
    
    return metrics


# ============================================================================
# Main Training Function
# ============================================================================

def train_unet(
    data_files: List[Path],
    output_dir: Path,
    loss_type: str = 'mse',
    gradient_weight: float = 0.5,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Train U-NET model.
    
    Args:
        data_files: List of spatial .npz files
        output_dir: Directory for outputs
        loss_type: 'mse' or 'mse_grad'
        gradient_weight: Weight for gradient loss (if using mse_grad)
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        early_stopping_patience: Patience for early stopping
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed
        device: Torch device
    
    Returns:
        Dictionary with training results
    """
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("=" * 70)
    logger.info("U-NET Training for KLAM_21 Prediction")
    logger.info("=" * 70)
    logger.info(f"Data files: {len(data_files)}")
    logger.info(f"Loss type: {loss_type}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # Load dataset
    dataset = KLAMSpatialDataset(data_files)
    
    # Get grid shape for model config
    sample_x, sample_y = dataset[0]
    _, H, W = sample_x.shape
    
    logger.info(f"Grid shape: {H} × {W}")
    logger.info("")
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Dataset splits: train={n_train}, val={n_val}, test={n_test}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    config = UNetConfig(
        input_channels=3,
        output_channels=6,
        base_channels=64,
        depth=4,
        dropout=0.1,
        input_height=H,
        input_width=W,
    )
    
    model = UNet(config).to(device)
    logger.info(model.summary())
    logger.info("")
    
    # Loss function
    criterion = get_loss_function(loss_type, gradient_weight).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch': [],
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    logger.info("Starting training...")
    logger.info("-" * 70)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch'].append(epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': asdict(config),
            }, output_dir / 'best_model.pth')
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
            )
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    logger.info("-" * 70)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model for evaluation
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    logger.info("")
    logger.info("Evaluating on test set...")
    test_metrics = compute_metrics(model, test_loader, device, dataset.OUTPUT_VARS)
    
    logger.info("Test metrics:")
    for var in dataset.OUTPUT_VARS:
        m = test_metrics[var]
        logger.info(f"  {var}: MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
    logger.info(f"  Overall: MSE={test_metrics['overall']['mse']:.4f}, "
                f"MAE={test_metrics['overall']['mae']:.4f}, "
                f"R²={test_metrics['overall']['r2']:.4f}")
    
    # Save results
    results = {
        'config': asdict(config),
        'training': {
            'loss_type': loss_type,
            'gradient_weight': gradient_weight if loss_type == 'mse_grad' else None,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'seed': seed,
        },
        'data': {
            'files': [str(f) for f in data_files],
            'n_total': n_total,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
        },
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'test_metrics': test_metrics,
    }
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save normalization parameters
    with open(output_dir / 'normalization.json', 'w') as f:
        norm_params = dataset.get_normalization_params()
        # Convert numpy types to Python types
        norm_params_clean = {}
        for key, val in norm_params.items():
            if isinstance(val, dict):
                norm_params_clean[key] = {}
                for k2, v2 in val.items():
                    if isinstance(v2, dict):
                        norm_params_clean[key][k2] = {k3: float(v3) for k3, v3 in v2.items()}
                    else:
                        norm_params_clean[key][k2] = float(v2)
            else:
                norm_params_clean[key] = float(val)
        json.dump(norm_params_clean, f, indent=2)
    
    logger.info("")
    logger.info(f"Results saved to {output_dir}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-NET for KLAM_21 spatial prediction"
    )
    
    # Data
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=['sail', 'random'],
        help="Type of training data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory for data (default: results/exp1_gp_training_data/sail_data or results/exp1_gp_training_data/random_data)"
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        default=60,
        help="Parcel size in meters (default: 60)"
    )
    
    # Model/Training
    parser.add_argument(
        "--loss",
        type=str,
        default='mse',
        choices=['mse', 'mse_grad'],
        help="Loss function"
    )
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.5,
        help="Weight for gradient loss (only used if loss=mse_grad)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    # Misc
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test with reduced epochs"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine data directory
    if args.data_dir is None:
        if args.data_type == 'sail':
            data_dir = Path("results/exp1_gp_training_data/sail_data")
        else:
            data_dir = Path("results/exp1_gp_training_data/random_data")
    else:
        data_dir = Path(args.data_dir)
    
    # Find data files
    parcel_size = args.parcel_size
    
    if args.data_type == 'sail':
        # SAIL: sail_27x27_rep1_spatial.npz, sail_27x27_rep2_spatial.npz, ...
        pattern = f"sail_{parcel_size}x{parcel_size}_rep*_spatial.npz"
        data_files = sorted(data_dir.glob(pattern))
    else:
        # Random: random_sobol_27m_n15400_seed2000_spatial.npz
        pattern = f"random_sobol_{parcel_size}m_*_spatial.npz"
        data_files = sorted(data_dir.glob(pattern))
    
    if len(data_files) == 0:
        logger.error(f"No data files found matching pattern: {data_dir / pattern}")
        logger.error(f"Available files in {data_dir}:")
        for f in sorted(data_dir.glob("*spatial.npz"))[:10]:
            logger.error(f"  {f.name}")
        return
    
    logger.info(f"Found {len(data_files)} data files:")
    for f in data_files:
        logger.info(f"  {f.name}")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = Path("results/unet_experiment") / f"{args.data_type}_{args.loss}_seed{args.seed}"
    else:
        output_dir = Path(args.output_dir)
    
    # Smoke test overrides
    epochs = args.epochs
    if args.smoke_test:
        epochs = 10
        logger.info("SMOKE TEST MODE: Running with 10 epochs")
    
    # Device
    device = torch.device('cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Train
    results = train_unet(
        data_files=data_files,
        output_dir=output_dir,
        loss_type=args.loss,
        gradient_weight=args.gradient_weight,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best R² (overall): {results['test_metrics']['overall']['r2']:.4f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
