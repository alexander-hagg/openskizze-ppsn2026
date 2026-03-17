# Experiment 6: QD Comparison with Offline Surrogates

**Research Question**: Can offline surrogate models (U-Net and SVGP) enable "coffee break" quality-diversity optimization (<15 min) with validated QD scores competitive with SAIL? How do different uncertainty-based exploration strategies affect final archive quality?

## Overview

This experiment compares MAP-Elites optimization using different offline surrogate configurations:

| Config | Model(s) | Fitness Function | Exploration Strategy | Parcel Size |
|--------|----------|------------------|---------------------|-------------|
| **1. U-Net Baseline** | U-Net | `f = U-Net(x)` | Pure exploitation | 27m |
| **2. SVGP Baseline** | SVGP | `f = SVGP_mean(x)` | Pure exploitation | 27m |
| **3. SVGP + UCB** | SVGP | `f_adj = SVGP_mean(x) + λ * SVGP_std(x)` | Uncertainty-based exploration | 27m |
| **4. U-Net + SVGP** | U-Net + SVGP | `f_adj = U-Net(x) + λ * SVGP_std(x)` | Accuracy + exploration | 27m |

**UCB λ values tested**: [0.1, 1.0, 10.0]

**Note**: All configurations use 27m parcels for fair comparison. The U-Net was trained on 27×27m parcels with 66×94 grid dimensions, which constrains the experiment to this parcel size.

## Directory Structure

```
experiments/exp6_qd_comparison/
├── run_mapelites_offline.py    # Main optimization script
├── validate_archives.py        # KLAM_21 validation
├── analyze_qd_comparison.py    # Result analysis
└── README.md                   # This file

hpc/exp6_qd_comparison/
├── submit_qd_optimization.sh   # Phase 1: Optimization (36 jobs)
├── submit_validation.sh        # Phase 2: KLAM_21 validation
├── submit_analysis.sh          # Phase 3: Analysis
└── run_exp6.sh                 # Master script

results/exp6_qd_comparison/
├── archive_*.pkl               # QD archives (from optimization)
├── metadata_*.json             # Run metadata
├── validation/                 # KLAM_21 validation results
│   ├── *_validated.npz         # Validated solutions
│   └── *_validated_metrics.json  # Validation metrics
└── analysis/                   # Comparative analysis
    ├── comparison_table.csv    # Summary table
    ├── qd_score_comparison.png # QD score plots
    ├── prediction_accuracy.png # Accuracy scatter plots
    ├── diversity_comparison.png # Diversity metrics
    └── ucb_lambda_effect.png   # UCB parameter sweep
```

## Running the Experiment

### Quick Start (HPC)

```bash
# Full pipeline
bash hpc/exp6_qd_comparison/run_exp6.sh all

# Individual phases
bash hpc/exp6_qd_comparison/run_exp6.sh opt        # Phase 1: Optimization
bash hpc/exp6_qd_comparison/run_exp6.sh validate   # Phase 2: Validation
bash hpc/exp6_qd_comparison/run_exp6.sh analyze    # Phase 3: Analysis

# Check status
bash hpc/exp6_qd_comparison/run_exp6.sh status
```

### Manual Execution

#### Phase 1: Optimization (~1 hour per config)

```bash
# Config 1: U-Net baseline (pure exploitation) - REQUIRES 27m parcels
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model unet \
    --parcel-size 27 \
    --generations 100000 \
    --seed 42

# Config 2: SVGP baseline (pure exploitation)
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model svgp \
    --parcel-size 27 \
    --generations 100000 \
    --seed 42

# Config 3: SVGP + UCB (exploration)
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model svgp \
    --ucb-lambda 1.0 \
    --parcel-size 51 \
    --generations 100000 \
    --seed 42

# Config 4: U-Net + SVGP hybrid (accuracy + exploration) - REQUIRES 27m parcels
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model hybrid \
    --ucb-lambda 1.0 \
    --parcel-size 27 \
    --generations 100000 \
    --seed 42
```

#### Phase 2: Validation (~5-10 hours per archive)

```bash
# Validate single archive
python experiments/exp6_qd_comparison/validate_archives.py \
    --archive results/exp6_qd_comparison/archive_unet_size27_seed42.pkl \
    --output-dir results/exp6_qd_comparison/validation

# Validate all archives
for archive in results/exp6_qd_comparison/archive_*.pkl; do
    python experiments/exp6_qd_comparison/validate_archives.py \
        --archive $archive \
        --output-dir results/exp6_qd_comparison/validation
done
```

#### Phase 3: Analysis (~5 minutes)

```bash
python experiments/exp6_qd_comparison/analyze_qd_comparison.py \
    --results-dir results/exp6_qd_comparison/validation \
    --output-dir results/exp6_qd_comparison/analysis
```

## Experimental Design

### Parameters

| Parameter | Value |
|-----------|-------|
| Algorithm | Pure MAP-Elites (no online updates) |
| Archive | 8D features × 5 bins = 390,625 cells |
| Generations | 100,000 |
| Emitters | 128 |
| Batch size | 8 |
| Total evaluations | ~100M |
| Parcel size | 51×51m |
| Replicates | 3 per configuration |
| UCB λ values | [0.1, 1.0, 10.0] |

### Model Paths (Default)

```yaml
U-Net: results/exp5_unet/unet_experiment/sail_mse_seed42/best_model.pth
SVGP:  results/exp3_hpo/hyperparameterization/model_optimized_ind2500_kmeans_rep1.pth
```

### Archive Storage

For uncertainty-based methods (Configs 3 & 4), each archive cell stores:

```python
{
    'solution': genome,               # 60D building parameters
    'measures': features,             # 8D behavioral features
    'objective_predicted': float,     # Model prediction (U-Net or SVGP_mean)
    'objective_adjusted': float,      # With exploration bonus (for selection)
    'svgp_uncertainty': float,        # SVGP stddev (for analysis)
    'objective_klam': float           # Ground truth (post-validation)
}
```

**Selection rule**: Solutions compete on `objective_adjusted`. If new solution has higher `objective_adjusted` than existing cell occupant, it replaces.

## Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **QD Score (Validated)** | Sum of true fitness values | `Σ objective_klam` |
| **Archive Coverage** | Fraction of cells filled | `n_occupied / 390,625` |
| **Phenotypic Diversity** | Solow-Polasky diversity | Based on genome distances |
| **Prediction Accuracy** | Model vs KLAM correlation | R², Spearman ρ, RMSE |
| **Wall Time** | Optimization duration | seconds |

## Hypotheses

1. **U-Net baseline** will achieve highest validated QD score due to superior prediction accuracy
2. **SVGP baseline** will achieve lower QD score due to prediction errors
3. **SVGP + UCB** will improve over SVGP baseline through uncertainty-driven exploration
4. **U-Net + SVGP** will achieve best validated QD score by combining U-Net accuracy with SVGP exploration
5. **All offline methods** will complete in <15 min (vs 580 hours for pure KLAM)
6. **Optimal λ** will be ~1.0 (balanced exploration-exploitation)

## Expected Results

### Optimization Time

- **Configs 1-2**: ~3-5 min (single model inference)
- **Config 3**: ~3-5 min (SVGP only)
- **Config 4**: ~5-7 min (U-Net + SVGP inference)

### Archive Quality (Predicted Ranking)

1. **U-Net + SVGP (λ=1.0)** - Best accuracy + exploration
2. **U-Net baseline** - High accuracy, less diversity
3. **SVGP + UCB (λ=1.0)** - Balanced but lower accuracy
4. **SVGP baseline** - Lowest accuracy

## Analysis Outputs

The analysis script generates:

1. **Comparison Table** (`comparison_table.csv`)
   - QD scores (predicted vs validated)
   - Prediction accuracy (R², RMSE, Spearman ρ)
   - Archive coverage and diversity
   - Wall times

2. **QD Score Comparison Plot** (`qd_score_comparison.png`)
   - Bar chart comparing predicted vs validated QD scores

3. **Prediction Accuracy Plot** (`prediction_accuracy.png`)
   - Scatter plots for each configuration (predicted vs KLAM)

4. **Diversity Comparison Plot** (`diversity_comparison.png`)
   - Mean pairwise distance and Solow-Polasky diversity

5. **UCB Lambda Effect Plot** (`ucb_lambda_effect.png`)
   - Effect of λ on QD score, accuracy, and coverage

## Key Questions

- Does uncertainty-based exploration improve validated QD scores?
- Is the added complexity of Config 4 (U-Net + SVGP) justified?
- Can we achieve 90%+ of SAIL's QD score in <1% of the time?
- What is the optimal UCB lambda value?

## Troubleshooting

### No archives found after optimization

```bash
# Check if jobs completed successfully
bash hpc/exp6_qd_comparison/run_exp6.sh status

# Check for errors in logs
tail logs/exp6_qd_*.err
```

### Validation taking too long

```bash
# Test with limited solutions first
python experiments/exp6_qd_comparison/validate_archives.py \
    --archive archive_unet_size51_seed42.pkl \
    --output-dir results/exp6_qd_comparison/validation \
    --max-solutions 100
```

### Analysis script fails

```bash
# Check if validation results exist
ls -lh results/exp6_qd_comparison/validation/*_validated.npz

# Check validation metrics
cat results/exp6_qd_comparison/validation/*_validated_metrics.json | head
```

## Notes

- **Validation is for experimental analysis only**, not part of the "coffee break" pipeline
- KLAM_21 evaluations take ~5-10 min each, so validating 1,000 solutions takes ~5-10 hours
- For production deployment, skip validation and use surrogate predictions directly
- Archive sizes vary by configuration (typically 1,000-10,000 solutions)

## References

- **MAP-Elites**: Mouret & Clune (2015). "Illuminating the space of possible solutions"
- **UCB**: Auer (2002). "Using confidence bounds for exploitation-exploration trade-offs"
- **SVGP**: Hensman et al. (2015). "Scalable Variational Gaussian Process Classification"
- **U-Net**: Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
