#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Experiment 7: Multi-Scale U-Net vs Single-Scale U-Nets

Research Question:
    Can a single multi-scale U-Net trained on mixed parcel sizes achieve
    comparable accuracy to parcel-specific single-scale U-Nets?

Approach:
    1. Train single-scale U-Nets on individual parcel sizes (27m, 51m, 69m)
    2. Train multi-scale U-Net on mixed data from all sizes
    3. Evaluate all models on all parcel sizes (generalization test)
    4. Compare accuracy, training time, model size

Expected Outcome:
    Multi-scale U-Net should achieve ~90%+ of single-scale performance
    with the advantage of handling arbitrary parcel sizes.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unet import UNet, UNetConfig
from experiments.exp7_multiscale_unet.multiscale_unet import create_multiscale_unet, MultiScaleUNetConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialKLAMDataset(Dataset):
    """Dataset for spatial KLAM data."""
    
    def __init__(self, data_path: Path):
        """Load spatial data from .npz file."""
        logger.info(f"Loading {data_path}")
        data = np.load(data_path)
        
        # Inputs
        self.terrain = torch.from_numpy(data['terrain']).float()
        self.buildings = torch.from_numpy(data['buildings']).float()
        self.landuse = torch.from_numpy(data['landuse']).float()
        
        # Outputs (use final timestamp only for now)
        self.uq = torch.from_numpy(data['uq'][:, -1]).float()  # (N, H, W)
        self.vq = torch.from_numpy(data['vq'][:, -1]).float()
        self.uz = torch.from_numpy(data['uz'][:, -1]).float()
        self.vz = torch.from_numpy(data['vz'][:, -1]).float()
        self.Ex = torch.from_numpy(data['Ex'][:, -1]).float()
        self.Hx = torch.from_numpy(data['Hx'][:, -1]).float()
        
        # Metadata
        self.parcel_size = int(data['parcel_size_cells'][0] * 3)  # Convert to meters
        self.n_samples = len(self.terrain)
        
        logger.info(f"  Loaded {self.n_samples} samples, parcel size: {self.parcel_size}m")
        logger.info(f"  Grid shape: {self.terrain.shape[1:]} (H×W)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Stack inputs: (3, H, W)
        x = torch.stack([self.terrain[idx], self.buildings[idx], self.landuse[idx]], dim=0)
        
        # Stack outputs: (6, H, W)
        y = torch.stack([
            self.uq[idx], self.vq[idx], self.uz[idx],
            self.vz[idx], self.Ex[idx], self.Hx[idx]
        ], dim=0)
        
        return x, y


def load_spatial_data(
    data_dir: Path,
    parcel_sizes: List[int],
    data_source: str = 'sail',
    replicate: int = 1
) -> List[Dataset]:
    """
    Load spatial data for specified parcel sizes.
    
    Args:
        data_dir: Base directory containing spatial data
        parcel_sizes: List of parcel sizes to load (e.g., [60, 120, 240])
        data_source: 'sail' or 'random'
        replicate: Which replicate to use (for SAIL data)
    
    Returns:
        List of datasets, one per parcel size
    """
    datasets = []
    
    for size in parcel_sizes:
        if data_source == 'sail':
            pattern = f"sail_{size}x{size}_rep{replicate}_spatial.npz"
            data_path = data_dir / 'sail_data' / pattern
        else:
            # Random data uses seed-based naming
            seed = 2000 + parcel_sizes.index(size)
            pattern = f"random_sobol_{size}m_n15400_seed{seed}_spatial.npz"
            data_path = data_dir / 'random_data' / pattern
        
        if not data_path.exists():
            logger.warning(f"Data not found: {data_path}")
            continue
        
        dataset = SpatialKLAMDataset(data_path)
        datasets.append(dataset)
    
    return datasets


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        
        all_preds.append(y_pred.cpu())
        all_targets.append(y.cpu())
    
    preds = torch.cat(all_preds, dim=0)  # (N, 6, H, W)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics per output field
    mse_per_field = torch.mean((preds - targets) ** 2, dim=(0, 2, 3))  # (6,)
    mae_per_field = torch.mean(torch.abs(preds - targets), dim=(0, 2, 3))
    
    # R² per field
    ss_res = torch.sum((targets - preds) ** 2, dim=(0, 2, 3))
    ss_tot = torch.sum((targets - torch.mean(targets, dim=(0, 2, 3), keepdim=True)) ** 2, dim=(0, 2, 3))
    r2_per_field = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Overall metrics
    mse_overall = torch.mean(mse_per_field).item()
    mae_overall = torch.mean(mae_per_field).item()
    r2_overall = torch.mean(r2_per_field).item()
    
    field_names = ['uq', 'vq', 'uz', 'vz', 'Ex', 'Hx']
    
    metrics = {
        'mse': mse_overall,
        'mae': mae_overall,
        'r2': r2_overall,
    }
    
    for i, name in enumerate(field_names):
        metrics[f'mse_{name}'] = mse_per_field[i].item()
        metrics[f'mae_{name}'] = mae_per_field[i].item()
        metrics[f'r2_{name}'] = r2_per_field[i].item()
    
    return metrics


def train_single_scale_unet(
    parcel_size: int,
    data_dir: Path,
    output_dir: Path,
    max_epochs: int = 200,
    batch_size: int = 32,
    device: torch.device = torch.device('cuda'),
    seed: int = 42
) -> Dict:
    """
    Train a single-scale U-Net on one parcel size.
    
    Returns:
        Dictionary with training history and best metrics
    """
    logger.info("="*80)
    logger.info(f"TRAINING SINGLE-SCALE U-NET: {parcel_size}m")
    logger.info("="*80)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    datasets = load_spatial_data(data_dir, [parcel_size], data_source='sail', replicate=1)
    if not datasets:
        raise ValueError(f"No data found for parcel size {parcel_size}m")
    
    dataset = datasets[0]
    grid_h, grid_w = dataset.terrain.shape[1:]
    
    # Train/val split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=seed)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Grid size: {grid_h}×{grid_w}")
    
    # Create model
    config = UNetConfig(
        input_channels=3,
        output_channels=6,
        base_channels=64,
        depth=4,
        dropout=0.1,
        input_height=grid_h,
        input_width=grid_w
    )
    model = UNet(config).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    patience = 20
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics['mse']
        val_r2 = val_metrics['r2']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            
            # Save best model
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"unet_single_{parcel_size}m_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'parcel_size': parcel_size,
                'grid_size': (grid_h, grid_w),
            }, model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val R²: {val_r2:.4f} | "
                f"Best R²: {best_metrics['r2']:.4f}"
            )
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    
    logger.info(f"Training complete in {elapsed/60:.1f} min")
    logger.info(f"Best validation R²: {best_metrics['r2']:.4f}")
    
    return {
        'parcel_size': parcel_size,
        'best_metrics': best_metrics,
        'history': history,
        'training_time': elapsed,
        'model_path': str(model_path),
    }


def train_multiscale_unet(
    parcel_sizes: List[int],
    data_dir: Path,
    output_dir: Path,
    max_epochs: int = 200,
    batch_size: int = 32,
    device: torch.device = torch.device('cuda'),
    seed: int = 42
) -> Dict:
    """
    Train a multi-scale U-Net on mixed parcel sizes.
    
    Returns:
        Dictionary with training history and best metrics per size
    """
    logger.info("="*80)
    logger.info(f"TRAINING MULTI-SCALE U-NET: {parcel_sizes}")
    logger.info("="*80)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data for all sizes
    datasets = load_spatial_data(data_dir, parcel_sizes, data_source='sail', replicate=1)
    if not datasets:
        raise ValueError("No datasets found")
    
    # Split each dataset into train/val
    train_datasets = []
    val_datasets = []
    
    for dataset in datasets:
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=seed)
        train_datasets.append(torch.utils.data.Subset(dataset, train_idx))
        val_datasets.append(torch.utils.data.Subset(dataset, val_idx))
    
    # Concatenate datasets
    train_dataset = ConcatDataset(train_datasets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create separate val loaders per size for evaluation
    val_loaders = {
        size: DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        for size, val_ds in zip(parcel_sizes, val_datasets)
    }
    
    total_train = sum(len(ds) for ds in train_datasets)
    logger.info(f"Total train samples: {total_train} (mixed sizes)")
    for size, val_ds in zip(parcel_sizes, val_datasets):
        logger.info(f"  Val samples ({size}m): {len(val_ds)}")
    
    # Create multi-scale model
    model = create_multiscale_unet(
        use_size_embedding=True,
        base_channels=64,
        depth=4,
        dropout=0.1
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_avg_r2 = -float('inf')
    best_metrics_per_size = {}
    patience = 20
    patience_counter = 0
    
    history = {'train_loss': [], 'val_metrics_per_size': {size: [] for size in parcel_sizes}}
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on each size separately
        val_metrics_per_size = {}
        for size, val_loader in val_loaders.items():
            metrics = evaluate(model, val_loader, device)
            val_metrics_per_size[size] = metrics
            history['val_metrics_per_size'][size].append(metrics)
        
        # Average R² across sizes
        avg_r2 = np.mean([m['r2'] for m in val_metrics_per_size.values()])
        avg_val_loss = np.mean([m['mse'] for m in val_metrics_per_size.values()])
        
        history['train_loss'].append(train_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_metrics_per_size = val_metrics_per_size
            patience_counter = 0
            
            # Save best model
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "unet_multiscale_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config,
                'parcel_sizes': parcel_sizes,
            }, model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Avg Val R²: {avg_r2:.4f} | "
                f"Best Avg R²: {best_avg_r2:.4f}"
            )
            for size in parcel_sizes:
                r2 = val_metrics_per_size[size]['r2']
                logger.info(f"  {size}m: R² = {r2:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    
    logger.info(f"Training complete in {elapsed/60:.1f} min")
    logger.info(f"Best average validation R²: {best_avg_r2:.4f}")
    for size, metrics in best_metrics_per_size.items():
        logger.info(f"  {size}m: R² = {metrics['r2']:.4f}")
    
    return {
        'parcel_sizes': parcel_sizes,
        'best_metrics_per_size': best_metrics_per_size,
        'best_avg_r2': best_avg_r2,
        'history': history,
        'training_time': elapsed,
        'model_path': str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 7: Multi-Scale U-Net Comparison")
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'multi', 'both'],
                       help='Training mode: single-scale, multi-scale, or both')
    parser.add_argument('--parcel-sizes', type=int, nargs='+',
                       default=[60, 120, 240],
                       help='Parcel sizes to train on (in meters)'))
    parser.add_argument('--data-dir', type=str,
                       default='/home/ahagg2s/openskizze-klam21-optimization/results/exp1_gp_training_data',
                       help='Directory containing spatial data')
    parser.add_argument('--output-dir', type=str,
                       default='results/exp7_multiscale_unet',
                       help='Output directory')
    parser.add_argument('--max-epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Device: {device}")
    logger.info(f"Parcel sizes: {args.parcel_sizes}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    results = {}
    
    # Train single-scale models
    if args.mode in ['single', 'both']:
        single_results = []
        for size in args.parcel_sizes:
            result = train_single_scale_unet(
                parcel_size=size,
                data_dir=data_dir,
                output_dir=output_dir,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed
            )
            single_results.append(result)
        
        results['single_scale'] = single_results
    
    # Train multi-scale model
    if args.mode in ['multi', 'both']:
        multi_result = train_multiscale_unet(
            parcel_sizes=args.parcel_sizes,
            data_dir=data_dir,
            output_dir=output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed
        )
        results['multi_scale'] = multi_result
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'training_results.json'
    
    # Convert numpy types to native Python for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
