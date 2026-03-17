# Experiment 7: Multi-Scale vs Single-Scale U-Net Comparison

## Research Question

**Can a single multi-scale U-Net trained on mixed parcel sizes achieve comparable accuracy to training separate single-scale U-Nets for each parcel size?**

## Motivation

- **Current approach**: Train one U-Net per parcel size (27m, 51m, 69m, etc.)
- **Problem**: Requires storing and managing multiple models
- **Alternative**: Train one multi-scale U-Net that handles variable input dimensions

**Key Question**: Does the flexibility of multi-scale architecture compromise accuracy compared to specialized single-scale models?

## Experimental Design

### Approaches

#### 1. Single-Scale U-Nets (Baseline)
- Train 13 separate U-Nets, one per parcel size: 27m, 33m, 39m, 45m, 51m, 57m, 63m, 69m, 75m, 81m, 87m, 93m, 99m
- Each model specialized for its grid dimensions
- Grid sizes range from (66×94) cells (27m) to (266×358) cells (99m)

#### 2. Multi-Scale U-Net (Test)
- Train 1 U-Net on mixed data from all 3 parcel sizes
- Model dynamically adapts to variable input dimensions
- Uses size embedding + adaptive pooling

### Multi-Scale Architecture

**Key Components:**

1. **Size Embedding**: 
   - Encodes grid dimensions (H, W) → 16D embedding
   - Broadcast to spatial dimensions and concatenated with input
   - Provides size-awareness to all layers

2. **Adaptive Pooling**:
   - Bottleneck uses `adaptive_avg_pool2d` to fixed 8×8 size
   - Enables consistent processing regardless of input size
   - Upsamples back to original dimensions

3. **Dynamic Resizing**:
   - Decoder uses `F.interpolate` to match skip connection sizes
   - Final output resized to original input dimensions

**Input**: (B, 3, H, W) - terrain, buildings, landuse  
**Output**: (B, 6, H, W) - uq, vq, uz, vz, Ex, Hx

### Training Data

**Source**: `/home/ahagg2s/openskizze-klam21-optimization/results/exp1_gp_training_data/sail_data/`

| Parcel Size Range | Grid Size Range | Samples per Size | Total Samples |
|-------------------|-----------------|------------------|---------------|
| 27m - 99m (13 sizes) | (66×94) to (266×358) | ~9,000 (3 reps) | ~117,000 |

**All 13 sizes used**: 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99 meters

**Total Training Data**: ~117,000 samples (multi-scale) or ~9,000 per single-scale model

### Training Configuration

```yaml
# Common settings
max_epochs: 200
batch_size: 16  # Reduced for larger grids
learning_rate: 0.001
optimizer: Adam
loss_function: MSE
early_stopping_patience: 20

# Data split
train_split: 0.85
val_split: 0.15

# Learning rate schedule
lr_scheduler: ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10
```

### Evaluation Metrics

**Primary**:
- **R² Score**: Variance explained (overall and per-field)
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

**Per-Field Metrics**:
- uq: u-velocity at 2m height
- vq: v-velocity at 2m height
- uz: u-velocity column average
- vz: v-velocity column average
- Ex: Cold air content
- Hx: Cold air layer height

**Comparison**:
- Single-scale average R² vs multi-scale average R²
- Training time (total for 3 single-scale vs 1 multi-scale)
- Model parameters and disk size

## Running the Experiment

### Local Testing

```bash
# Test multi-scale U-Net architecture
cd experiments/exp7_multiscale_unet
python multiscale_unet.py

# Dry-run training (1 epoch)
python train_multiscale_comparison.py \
    --mode both \
    --parcel-sizes 27 51 69 \
    --data-dir /path/to/sail_data \
    --output-dir results/exp7_test \
    --max-epochs 1
```

### HPC Execution

```bash
cd hpc/exp7_multiscale_unet

# Submit all jobs (3 single + 1 multi = 4 jobs)
bash run_exp7.sh all

# Or submit individually
bash run_exp7.sh single  # 3 single-scale jobs
bash run_exp7.sh multi   # 1 multi-scale job

# Check status
bash run_exp7.sh status

# Run analysis when complete
bash run_exp7.sh analyze
```

### Expected Runtime

| Job Type | Time per Job | Resources | Total Jobs |
|----------|--------------|-----------|------------|
| Single-scale (small: 27-45m) | ~2-3 hours | 1 GPU, 32GB RAM | 4 jobs |
| Single-scale (medium: 51-69m) | ~3-5 hours | 1 GPU, 32GB RAM | 4 jobs |
| Single-scale (large: 75-99m) | ~5-6 hours | 1 GPU, 32GB RAM | 5 jobs |
| Multi-scale (all 13 sizes) | ~10-12 hours | 1 GPU, 128GB RAM | 1 job |

**Total sequential time**: ~50-65 hours (all single-scale models)  
**Total parallel time**: ~10-12 hours (with sufficient GPUs)

## Expected Results

### Hypothesis 1: Accuracy Trade-off
**Prediction**: Single-scale models will achieve slightly higher R² (~1-3% better) due to specialization.

**Rationale**: 
- Single-scale models optimized for fixed grid dimensions
- No need to generalize across sizes
- Expected: R² (single) ≈ 0.997, R² (multi) ≈ 0.99

### Hypothesis 2: Training Efficiency
**Prediction**: Multi-scale training time ≈ 1.5× single model, saving 50% vs training all 3.

**Rationale**:
- Multi-scale trains on 3× data (~27K samples vs ~9K)
- More complex optimization landscape
- Expected: Single total ~10 hours, Multi ~5-6 hours

### Hypothesis 3: Deployment Benefits
**Prediction**: Multi-scale offers 67% storage reduction (1 model vs 3).

**Advantages**:
- Simpler deployment (single checkpoint)
- Easier versioning and updates
- Handles intermediate sizes without retraining

## Analysis

After training, run:

```bash
python experiments/exp7_multiscale_unet/analyze_results.py \
    --results-file results/exp7_multiscale_unet/training_results.json \
    --output-dir results/exp7_multiscale_unet/analysis
```

**Outputs**:
- `comparison_table.csv` - Quantitative comparison
- `r2_comparison.png` - R² scores per parcel size
- `per_field_accuracy_51m.png` - Per-field breakdown
- `training_time.png` - Training time comparison
- `summary_report.txt` - Text summary

## Interpreting Results

### Scenario 1: Multi-Scale Competitive (R² > 0.99)
**Conclusion**: Use multi-scale model in production
- Simpler deployment
- Lower storage overhead
- Good enough accuracy

### Scenario 2: Multi-Scale Degraded (R² < 0.98)
**Conclusion**: Keep single-scale models
- Accuracy loss unacceptable for physics surrogate
- Maintain separate models per size
- Consider hybrid approach (multi-scale for exploration, single-scale for validation)

### Scenario 3: Mixed Results (Some sizes good, others poor)
**Conclusion**: Investigate architecture improvements
- Size embedding may need more capacity
- Try attention mechanisms for size-specific features
- Consider hierarchical approach (coarse multi-scale + fine-tuning)

## Files

```
experiments/exp7_multiscale_unet/
├── README.md                        # This file
├── multiscale_unet.py               # Multi-scale U-Net architecture
├── train_multiscale_comparison.py   # Training script
└── analyze_results.py               # Analysis and visualization

hpc/exp7_multiscale_unet/
├── run_exp7.sh                      # Master pipeline script
├── submit_train_single.sh           # SLURM array job (3 single-scale)
└── submit_train_multi.sh            # SLURM job (1 multi-scale)

results/exp7_multiscale_unet/
├── training_results.json            # All training metrics
├── model_single_27m.pt              # Single-scale checkpoints
├── model_single_51m.pt
├── model_single_69m.pt
├── model_multiscale.pt              # Multi-scale checkpoint
└── analysis/                        # Plots and tables
    ├── comparison_table.csv
    ├── r2_comparison.png
    ├── per_field_accuracy_51m.png
    ├── training_time.png
    └── summary_report.txt
```

## Future Work

### Dual U-Net Strategy (Production Deployment)

The current experiments compare single-scale vs multi-scale on a simplified 66×94 grid domain. For production deployment, a **dual U-Net strategy** will integrate:

1. **Small Generic U-Net** (current models)
   - Fast predictions (~2ms)
   - Generic boundary conditions
   - Used during QD optimization

2. **Large Site-Specific U-Net** (future work)
   - Trained on full-city KLAM_21 simulations
   - Real topography and distant cold air sources
   - Complex boundary conditions
   - Used for final validation

This dual approach enables fast generic optimization with high-fidelity site-specific validation.

## References

- **Original U-Net**: Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- **Adaptive Pooling**: He et al. (2015). Spatial Pyramid Pooling in Deep Convolutional Networks.
- **Size Embeddings**: Inspired by positional encodings in Transformers (Vaswani et al., 2017).
