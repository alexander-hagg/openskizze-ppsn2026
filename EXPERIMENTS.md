# Experiments and Results

This document describes the experiments conducted in the OpenSKIZZE KLAM-21 optimization project and their key findings.

---

## Table of Contents

1. [Experiment Overview](#experiment-overview)
2. [Experiment 1: GP Training Data Comparison](#experiment-1-gp-training-data-comparison)
3. [Experiment 2: Flux Sensitivity Analysis](#experiment-2-flux-sensitivity-analysis)
4. [Experiment 3: Hyperparameter Optimization](#experiment-3-hyperparameter-optimization)
5. [Experiment 4: MAP-Elites with Offline GP](#experiment-4-map-elites-with-offline-gp)
6. [Experiment 5: U-Net Deep Learning Surrogate](#experiment-5-u-net-deep-learning-surrogate)
7. [Experiment 6a: SVGP vs U-Net Model Comparison](#experiment-6a-svgp-vs-u-net-model-comparison)
8. [Experiment 6b: QD with Offline Surrogates](#experiment-6b-qd-with-offline-surrogates)
9. [Experiment 7: Multi-Scale U-Net Comparison](#experiment-7-multi-scale-u-net-comparison)
10. [Experiment 8: Performance Benchmarking](#experiment-8-performance-benchmarking)
11. [Experiment 9: KLAM_21 Boundary Analysis](#experiment-9-klam_21-boundary-analysis)
12. [Summary of Key Findings](#summary-of-key-findings)

---

## Experiment Overview

| Experiment | Research Question | Key Finding |
|------------|-------------------|-------------|
| **GP Data Comparison** | SAIL vs Random training data for GP? | Optimized-trained generalizes best |
| **Flux Sensitivity** | Which morphology parameters affect cold air flux? | GRZ dominates (ρ = -0.948) |
| **HPO** | Optimal SVGP hyperparameters? | 2000 inducing points optimal |
| **MAP-Elites GP** | Can offline GP replace KLAM in QD? | *In progress* |
| **U-Net Surrogate** | Can CNN predict spatial KLAM fields? | R² = 0.997 achieved |
| **Model Comparison (6a)** | SVGP vs U-Net: which surrogate for GUI? | *Ready to run* |
| **QD Offline Surrogates (6b)** | U-Net vs SVGP vs Hybrid for fast QD? | *Running* |
| **Multi-Scale U-Net** | One model for all sizes vs multiple? | *Pending* |
| **Performance Benchmark** | What are the QD optimization bottlenecks? | *Running* |
| **Boundary Analysis** | Landuse boundary artifacts in KLAM_21? | *Ready to run* |

---

## Experiment 1: GP Training Data Comparison

**Report**: [EXPERIMENT_PLAN_GP_TRAINING.md](EXPERIMENT_PLAN_GP_TRAINING.md)  
**Results**: [results/gp_evaluation/](results/gp_evaluation/)

### Research Question

Does training a GP surrogate on SAIL-optimized solutions (high-fitness region) generalize well, or do we need random samples to cover the full fitness landscape?

### Experimental Design

#### Training Datasets

| Dataset | Source | Size | Fitness Distribution |
|---------|--------|------|---------------------|
| **Optimized** | SAIL archives | ~26,000 | High fitness (optimized) |
| **Random** | Sobol sequence | ~26,000 | Full range (mostly low) |
| **Combined** | 50% each | ~26,000 | Mixed |

#### Parcel Sizes

13 sizes from 27m to 99m (6m increments):
```
27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99 meters
```

#### Evaluation

Cross-domain evaluation matrix (each model tested on all test sets):

```
                    Test-Optimized  Test-Random  Test-Combined
Model-Optimized         ✓ (ID)         ✓ (OOD)      ✓
Model-Random            ✓ (OOD)        ✓ (ID)       ✓
Model-Combined          ✓              ✓            ✓ (ID)
```
ID = In-Distribution, OOD = Out-of-Distribution

### Key Results

#### Cross-Domain Performance (R²)

| Model \ Eval Set | Optimized | Random | Combined | **Avg** |
|------------------|-----------|--------|----------|---------|
| **Optimized** | 0.839 | 0.680 | 0.762 | **0.760** |
| Random | -0.550 | 0.814 | 0.076 | 0.113 |
| Combined | 0.378 | 0.647 | 0.506 | 0.510 |

#### Key Finding: Optimized-Trained Generalizes Best

1. **Model-Optimized** achieves best average performance across domains
2. **Model-Random fails catastrophically** on optimized data (R² = -0.55)
3. **Model-Combined** doesn't provide "best of both worlds"

#### Why This Happens

- **Optimized data** spans the full fitness range through exploration
- SAIL's UCB acquisition explores low-fitness regions for diversity
- Random sampling concentrates in low-fitness regions (random designs are usually bad)
- The GP trained on optimized data learns the full landscape

### Implications

- **Use SAIL archives for GP training**, not random samples
- SAIL exploration + exploitation naturally covers the fitness landscape
- No need for explicit random augmentation

---

## Experiment 2: Flux Sensitivity Analysis

**Report**: [reports/flux_sensitivity/flux_sensitivity_report.pdf](reports/flux_sensitivity/flux_sensitivity_report.pdf)  
**Results**: [results/flux_sensitivity_v2/](results/flux_sensitivity_v2/)

### Research Questions

1. How does site coverage (GRZ) affect cold air flux?
2. Does building height significantly impact ground-level airflow?
3. Does building orientation relative to wind direction matter?

### Experimental Design

**Factorial design** with 3 factors:

| Factor | Levels | Values |
|--------|--------|--------|
| Site Coverage (GRZ) | 6 | 0%, 20%, 40%, 60%, 80%, 100% |
| Building Height | 5 | 0, 2, 4, 6, 8 floors (×3m) |
| Orientation | 3 | 0°, 45°, 90° to wind |

**Total**: 65 valid configurations (reduced from 90 due to constraints)

**Parcel**: 51×51m (17×17 cells at 3m resolution)

### Key Results

#### 1. GRZ is the Dominant Factor

| Factor | Spearman ρ | ANOVA F | p-value |
|--------|------------|---------|---------|
| **GRZ** | **-0.948** | **386.6** | **< 10⁻⁴³** |
| Height | -0.024 | 0.66 | 0.621 |
| Orientation | -0.035 | 0.08 | 0.920 |

**Interpretation**: 
- Every 20% increase in GRZ reduces cold air flux by ~15-20%
- Building height has no significant effect at ground level (2m)
- Orientation is negligible for bar layouts

#### 2. Physical Explanation

- Cold air drains around buildings, not through them
- Height matters for upper-level airflow but not pedestrian-level ventilation
- GRZ directly controls the flow cross-section

#### 3. Implications

- **For optimization**: GRZ is the primary design lever for cold air ventilation
- **For GP surrogate**: Model should capture GRZ→flux relationship accurately
- **For urban planning**: Site coverage regulations directly impact climate adaptation

### Figures

| Figure | Description |
|--------|-------------|
| `flux_heatmaps.png` | GRZ × Height flux response surface |
| `airflow_vectors.png` | Wind field visualizations |
| `feature_correlations.png` | Correlation matrix |
| `layout_thumbnails.png` | Example building configurations |

---

## Experiment 3: Hyperparameter Optimization

**Report**: [reports/hpo/hpo_report.pdf](reports/hpo/hpo_report.pdf)  
**Results**: [results/hyperparameterization/](results/hyperparameterization/)

### Research Question

What are the optimal hyperparameters for the SVGP surrogate model?

### Experimental Design

#### Search Space

| Parameter | Values |
|-----------|--------|
| Inducing Points | 500, 1000, 2000 |
| K-means Init | True, False |

**Total**: 6 configurations × 3 datasets × 3 replicates = 54 runs

#### Fixed Settings

| Parameter | Value |
|-----------|-------|
| Max Epochs | 200 |
| Early Stopping | 20 epochs patience |
| LR Warmup | 10 epochs |
| Batch Size | 1024 |
| Learning Rate | 0.01 |
| Kernel | Matérn 2.5 + ARD (62D) |

### Key Results

#### 1. Inducing Points: More is Better

| Inducing Points | Optimized R² | Random R² | Combined R² |
|-----------------|--------------|-----------|-------------|
| 500 | 0.907 | 0.840 | 0.503 |
| 1000 | 0.919 | 0.843 | 0.503 |
| **2000** | **0.930** | **0.846** | **0.503** |

**Effect size**: +2.5% R² improvement from 500→2000 for optimized data

#### 2. K-means Initialization: Negligible Effect

| Init Method | Optimized R² | Random R² | Combined R² |
|-------------|--------------|-----------|-------------|
| Random | 0.918 | 0.843 | 0.503 |
| K-means | 0.919 | 0.843 | 0.503 |

**Effect size**: <0.1% difference

#### 3. Best Configuration

```yaml
num_inducing: 2000
init_method: kmeans  # Marginal benefit
early_stopping: true
warmup_epochs: 10
```

#### 4. Cross-Domain HPO Results

Using best config (2000 inducing, K-means):

| Training Data | → Optimized | → Random | → Combined | Avg |
|---------------|-------------|----------|------------|-----|
| **Optimized** | **0.934** | 0.668 | **0.816** | **0.806** |
| Random | -0.293 | 0.850 | 0.426 | 0.328 |
| Combined | 0.000 | 0.767 | 0.504 | 0.424 |

### Implications

- **Use 2000 inducing points** (computational cost acceptable)
- K-means initialization provides negligible benefit
- Early stopping is essential (prevents overfitting)
- Optimized training data remains best choice

---

## Experiment 4: MAP-Elites with Offline GP

**Documentation**: [experiments/MAPELITES_GP_README.md](experiments/MAPELITES_GP_README.md)  
**Results**: [results/mapelites_gp/](results/mapelites_gp/)

### Research Question

Can a pre-trained GP surrogate completely replace KLAM_21 physics simulation during MAP-Elites optimization, enabling fast QD search for building layouts?

### Experimental Design

#### Approach

1. Train SVGP surrogate on SAIL archive data (using best HPO config)
2. Run pure MAP-Elites optimization using GP predictions only
3. Validate top archive solutions with real KLAM_21
4. Compare archive quality vs SAIL (which uses online surrogate updates)

#### Parameter Sweep

| Parameter | Values |
|-----------|--------|
| Generations | 10k, 50k, 100k |
| Batch Size | 32, 64, 128, 256, 512 |
| Parcel Size | 51m, 69m, 87m |

**Total**: 5 × 5 × 3 = 75 configurations

#### Metrics

- Archive coverage (% of cells filled)
- QD score (sum of fitness in archive)
- Prediction fidelity (GP vs KLAM_21 on validated solutions)
- Ranking correlation (Spearman ρ)

### Current Status

*Experiment in progress.* Initial results suggest:
- GP surrogate enables ~1000× faster evaluation
- Archive coverage comparable to SAIL
- Validation needed to confirm fitness accuracy

---

## Experiment 5: U-Net Deep Learning Surrogate

**Script**: [experiments/exp5_unet/train_unet_klam.py](experiments/exp5_unet/train_unet_klam.py)  
**Results**: [results/exp5_unet/unet_experiment/](results/exp5_unet/unet_experiment/)

### Research Question

Can a U-Net CNN learn to predict full KLAM_21 spatial output fields (velocity, cold air content) from building layout inputs, enabling both scalar fitness and spatial field prediction?

### Experimental Design

#### Architecture

- **Input**: 3-channel image (terrain, buildings, landuse)
- **Output**: 6-channel image (uq, vq, uz, vz, Ex, Hx at final timestep)
- **Model**: U-Net with ResNet encoder

#### Training Data

Uses spatial data from SAIL archives (generated with `--collect-spatial-data`):
- **SAIL data**: `results/exp1_gp_training_data/sail_data/sail_{size}x{size}_rep*_spatial.npz`
- **Random data**: `results/exp1_gp_training_data/random_data/random_sobol_{size}m_*_spatial.npz`
- Full KLAM_21 simulation fields saved per sample
- Input/output grids aligned

#### Evaluation Metrics

- Per-field MSE and MAE
- Derived fitness correlation (compute flux from predicted fields)
- Spatial pattern accuracy (SSIM)

### Results

#### Best Model Performance

**Configuration**: Small U-Net (64 base channels, depth=4, MSE loss)  
**Training Data**: 27,174 samples from SAIL archives (parcel size 27×27m)  

**Overall Test Performance**:
- **R² = 0.9973** (99.73% variance explained)
- **MSE = 0.0027** (normalized)
- **MAE = 0.0207** (normalized)

#### Per-Field Accuracy

| Output Field | MSE | MAE | R² | Description |
|--------------|-----|-----|----|----|
| **Ex** (Cold air content) | 0.000366 | 0.0115 | **0.9996** | Energy content |
| **Hx** (Cold air height) | 0.000350 | 0.0106 | **0.9997** | Layer thickness |
| **uq** (u-velocity @ 2m) | 0.000186 | 0.0090 | **0.9998** | Horizontal wind |
| **vq** (v-velocity @ 2m) | 0.0084 | 0.0377 | **0.9916** | Vertical wind |
| **uz** (u-velocity avg) | 0.000620 | 0.0201 | **0.9994** | Column-averaged |
| **vz** (v-velocity avg) | 0.0064 | 0.0356 | **0.9936** | Column-averaged |

**Training Details**:
- Best epoch: 74 out of 200 (early stopping)
- Best validation loss: 0.00275
- Total training time: ~1.27 hours on NVIDIA A100

**Throughput**:
- Optimal batch size: 128
- ~465 samples/second on NVIDIA A100
- ~2ms per prediction

### Future Work: Dual U-Net Strategy

The current "small" U-Net model is trained on a simplified 66×94 grid simulation box with generic boundary conditions (flat terrain, west wind, no neighboring buildings). 

**Planned Enhancement**: A dual U-Net architecture will integrate:
1. **Small Generic U-Net** (current model) - Fast predictions for standard scenarios
2. **Large Site-Specific U-Net** - Trained on full-city simulations capturing:
   - Real topography and terrain features
   - Cold air sources from distant areas (parks, forests)
   - Complex boundary conditions
   - Neighboring building effects

This dual strategy will enable both fast generic optimization and high-fidelity site-specific validation.

---

## Experiment 6a: SVGP vs U-Net Model Comparison

**Documentation**: [experiments/exp6_model_comparison/README.md](experiments/exp6_model_comparison/README.md)  
**Script**: [experiments/exp6_model_comparison/compare_svgp_unet.py](experiments/exp6_model_comparison/compare_svgp_unet.py)  
**Results**: [results/model_comparison/](results/model_comparison/)

### Research Question

Which offline surrogate model — SVGP or U-Net — should be used for MAP-Elites optimization in the GUI? This is a head-to-head comparison on identical test layouts.

### Experimental Design

#### Models Compared

| Model | Input | Output | R² | Key Property |
|-------|-------|--------|----|--------------|
| **SVGP** | 62D (60D genome + width + height) | Scalar fitness + uncertainty | 0.93 | Calibrated uncertainty |
| **U-Net** | 66×94 grid (terrain, buildings, landuse) | 6 spatial fields → scalar fitness | 0.997 | Full spatial prediction |

#### Test Data

- **1000 diverse layouts** generated via Sobol quasi-random sequence
- Genomes scaled to [-1, 1], evaluated at fixed parcel size (default 51m)
- Both models predict fitness for the identical set of layouts

#### Comparison Metrics

1. **Agreement**: Pearson r (linear), Spearman ρ (rank — most important for QD)
2. **Ranking Fidelity**: Top-100 overlap between models
3. **Throughput**: Samples/second on GPU
4. **Calibration**: RMSE, fitness distribution match, Bland-Altman analysis

### Decision Criteria

- **If ρ > 0.85 AND U-Net throughput > 10× SVGP** → Use U-Net
- **If ρ > 0.90 AND similar throughput** → Use SVGP (more mature)
- **Else** → Use both (ensemble) or choose based on integration ease

### Running the Experiment

```bash
# Quick local test (100 samples)
bash experiments/exp6_model_comparison/test_comparison.sh

# Full HPC run (1000 samples)
sbatch hpc/exp6_model_comparison/submit_comparison.sh
```

### Output Files

All saved to `results/model_comparison/`:
- `svgp_vs_unet_comparison.json` — Metrics, recommendation, throughput comparison
- `comparison_plots.png` — Scatter, residual, distribution, ranking, Bland-Altman plots
- `predictions.npz` — Raw predictions from both models for further analysis

### Expected Results

- **Spearman ρ**: 0.85–0.95 (high agreement expected)
- **U-Net throughput**: ~1000–1200 samples/s
- **SVGP throughput**: ~200–500 samples/s
- **Throughput ratio**: 2–5× (U-Net faster)

### Current Status

*Ready to run. Est. time: 30–60 minutes on HPC with GPU.*

---

## Experiment 6b: QD with Offline Surrogates

**Documentation**: [experiments/exp6_qd_comparison/](experiments/exp6_qd_comparison/)  
**Results**: [results/exp6_qd_comparison/](results/exp6_qd_comparison/)

### Research Question

Can offline surrogate models (U-Net and SVGP) enable "coffee break" quality-diversity optimization (<15 min) that produces archives whose solutions are validated as high-quality by the real KLAM_21 simulator? Is the SVGP needed at all, or does the U-Net alone suffice?

### Motivation

- **U-Net** (Exp 5): Highest spatial prediction accuracy (R² ≈ 0.997 on grids) but provides no uncertainty estimates
- **SVGP** (Exp 3): Lower accuracy (R² ≈ 0.93 on scalar fitness) but provides calibrated uncertainty
- **SAIL** (online GP): Proven but extremely slow — the GP surrogate is retrained online during optimization
- **Key question**: We trained the U-Net on GP+SAIL archive data (Exps 1-5). Can we now drop the online GP loop entirely and just run MAP-Elites with the U-Net?

### Experimental Design

#### Configurations (8 approaches × 3 seeds)

| Config | Model(s) | Fitness Function | Exploration | UCB λ |
|--------|----------|------------------|-------------|-------|
| **unet** | U-Net only | `f = UNet(x)` | Pure exploitation | 0.0 |
| **svgp** | SVGP only | `f = SVGP_mean(x)` | Pure exploitation | 0.0 |
| **svgp_ucb0.1** | SVGP | `f = mean + 0.1·std` | Mild exploration | 0.1 |
| **svgp_ucb1.0** | SVGP | `f = mean + 1.0·std` | Moderate exploration | 1.0 |
| **svgp_ucb10.0** | SVGP | `f = mean + 10·std` | Aggressive exploration | 10.0 |
| **hybrid_ucb0.1** | U-Net + SVGP | `f = UNet(x) + 0.1·std` | Mild exploration | 0.1 |
| **hybrid_ucb1.0** | U-Net + SVGP | `f = UNet(x) + 1.0·std` | Moderate exploration | 1.0 |
| **hybrid_ucb10.0** | U-Net + SVGP | `f = UNet(x) + 10·std` | Aggressive exploration | 10.0 |

**Seeds**: 42, 43, 44 (3 replicates each → **24 optimization jobs total**)

#### Optimization Settings

| Parameter | Value |
|-----------|-------|
| Algorithm | MAP-Elites (offline surrogates, no online retraining) |
| Archive | 8D GridArchive × 5 bins per dim = 390,625 cells |
| Generations | 5,000 |
| Emitters | 64 GaussianEmitters |
| Batch size | 8 |
| Parcel size | 60m (20×20 cells) |
| Domain grid | 66×89 (matching KLAM rectangular domain) |
| Replicates | 3 per configuration |
| GPU | 1× (A100 or similar) |

#### Validation Phase

After optimization, 100 solutions per archive are validated against real KLAM_21:
- **Stratified sampling**: One solution per occupied archive cell (maximizes diversity)
- **Ground truth**: Full KLAM_21 physics simulation per solution
- **Matched comparison**: QD scores computed over the *same* 100 solutions for both predicted and validated objectives

### Results (Run 1 — with domain construction bias)

Initial results revealed a clear model hierarchy, though a systematic ~2× scale bias was present (see [Domain Construction Bug](#domain-construction-bug) below).

#### Model Ranking by Prediction Quality

| Tier | Model | Spearman ρ | QD Ratio | R² | RMSE |
|------|-------|-----------|----------|-----|------|
| **A** | **unet** (λ=0) | **0.960–0.970** | **2.02** | -13 to -20 | 14.8–15.3 |
| **A** | **hybrid_ucb0.1** | **0.954–0.972** | **1.97** | -12 to -19 | 14.6–14.9 |
| B | hybrid_ucb1.0 | 0.755–0.808 | 1.45 | -4 to -6 | 9.1–9.5 |
| C | hybrid_ucb10.0 | 0.543–0.559 | 0.34 | -165 to -280 | 55.8–57.2 |
| D | svgp_ucb0.1 | 0.236–0.544 | 0.61 | -19 to -25 | 18.0–19.5 |
| D | svgp (λ=0) | 0.307–0.425 | 0.61 | -20 to -23 | 18.0–18.9 |
| D | svgp_ucb1.0 | 0.309–0.341 | 0.56 | -38 to -40 | 23.3–23.9 |
| E | svgp_ucb10.0 | -0.009–0.207 | 0.25 | -480 to -542 | 83.9–86.5 |

*QD Ratio = QD_validated / QD_predicted (matched subset). Values >1 indicate U-Net underestimates; <1 indicates overestimation.* The R² is negative because of a systematic scale offset (not random error) — ranking preservation (ρ) is the relevant metric for MAP-Elites.

#### Key Findings

**1. U-Net alone is sufficient — SVGP adds no value.**
The pure U-Net baseline (ρ ≈ 0.965) is statistically indistinguishable from hybrid_ucb0.1 (ρ ≈ 0.963). Adding SVGP uncertainty with λ=0.1 provides no measurable benefit. This is the central finding: **the online GP is no longer needed.**

**2. SVGP alone is inadequate for QD optimization.**
All pure-SVGP configurations achieve ρ < 0.55 — insufficient for reliable ranking of candidate solutions. The SVGP was designed for scalar fitness prediction from 62D genome inputs, whereas the U-Net operates on spatial grids and captures the physics much better.

**3. High UCB λ is counterproductive.**
λ=10.0 degrades all models severely (ρ drops to 0.03–0.56). The SVGP uncertainty estimates are not well-calibrated enough to drive meaningful exploration at high weights. Even moderate exploration (λ=1.0) hurts performance (ρ drops to 0.31–0.81).

**4. Bootstrapping from SAIL to offline MAP-Elites works.**
The research pipeline was: (1) SAIL with online GP surrogate → generates training data → (2) train U-Net on SAIL archives → (3) offline MAP-Elites with frozen U-Net. This experiment validates step 3: the U-Net, trained purely on GP+SAIL data, enables standalone QD optimization *without any online surrogate updates*. The SAIL algorithm has effectively been bootstrapped away.

**5. Systematic 2× scale bias indicates a domain construction bug.**
The QD ratio is consistently ~2.0 for the best models (KLAM fitness is 2× the U-Net prediction). Ranking is preserved (ρ > 0.95), but the absolute scale is off. This was traced to mismatches in the domain grid construction between the U-Net evaluator and the KLAM reference (see below).

#### Domain Construction Bug

Investigation revealed three discrepancies in `optimized_construct_domain_grids()` relative to KLAM's `compute_fitness_klam()`:

| Property | KLAM (ground truth) | U-Net evaluator (was) | Impact |
|----------|--------------------|-----------------------|--------|
| **Parcel x-offset** | col 46 (shifted right by left_extension) | col 34 (naively centered) | Buildings in wrong position |
| **Terrain** | 2° katabatic slope | Flat (all zeros) | No gravitational forcing signal |
| **Landuse** | 7 left of parcel, 2 from parcel rightward | 7 everywhere except building cells | Wrong surface type classification |

The KLAM domain extends the grid leftward (upwind) by `left_extension = (env_cells_base - parcel_cells) // 2` cells, placing the parcel at a non-centered position. The original evaluator code naively centered the parcel in the full grid.

**Fix applied**: `optimized_construct_domain_grids()` and `compute_klam_roi_mask()` now replicate the exact KLAM domain conventions (slope, landuse pattern, parcel offset). **Re-run required** to obtain corrected results.

### Implications

**For the OpenSKIZZE system**: The U-Net surrogate, trained on ~2000 KLAM evaluations from SAIL archives, enables fully offline MAP-Elites optimization. The original SAIL pipeline (which required expensive online GP retraining every generation) can be replaced by a single U-Net forward pass per candidate. This reduces optimization time from hours/days (SAIL with KLAM) to minutes (MAP-Elites with U-Net).

**For surrogate-assisted QD**: The SVGP's uncertainty-based exploration (UCB) provided no benefit when the U-Net already has sufficient accuracy. This suggests that for well-trained deep surrogates, pure exploitation can outperform exploration-exploitation trade-offs — the surrogate landscape is faithful enough that greedy optimization finds good solutions.

**For the GUI**: The U-Net model can be deployed standalone (no SVGP dependency) for interactive design exploration, simplifying the inference stack.

### Running the Experiment

```bash
# Full experiment (optimization → validation → analysis)
bash hpc/exp6_qd_comparison/run_exp6.sh all

# Individual phases
bash hpc/exp6_qd_comparison/run_exp6.sh opt        # 24 GPU jobs
bash hpc/exp6_qd_comparison/run_exp6.sh validate    # 24 CPU jobs (100 KLAM evals each)
bash hpc/exp6_qd_comparison/run_exp6.sh analyze     # Local analysis + plots
```

### Current Status

**Run 1**: Complete (24 configs × 3 seeds). Results show clear U-Net dominance. Systematic 2× scale bias identified and fixed in domain construction code.  
**Run 2**: Pending — re-run with corrected domain grids (terrain slope, landuse, parcel position) to verify scale bias is eliminated.

---

## Experiment 7: Multi-Scale U-Net Comparison

**Documentation**: [experiments/exp7_multiscale_unet/README.md](experiments/exp7_multiscale_unet/README.md)  
**Results**: [results/exp7_multiscale_unet/](results/exp7_multiscale_unet/)

### Research Question

Can a single multi-scale U-Net trained on mixed parcel sizes achieve comparable accuracy to training separate single-scale U-Nets for each parcel size?

### Motivation

**Current Approach**: Train one specialized U-Net per parcel size (all 13: 27–99m)
- ✓ High accuracy (R² = 0.997 per size)
- ✗ 13 models to manage
- ✗ 13× storage overhead (~650MB total)
- ✗ Requires retraining for new sizes

**Alternative**: Train one multi-scale U-Net on mixed data
- ? Potentially lower accuracy (target: R² > 0.99)
- ✓ Single model deployment (~50MB)
- ✓ 92% storage reduction (1 model vs 13)
- ✓ Handles all sizes + intermediate values
- ✓ 80% faster training (12 hours vs 50 hours)

### Experimental Design

#### Configurations

| Configuration | Models | Training Data | Grid Sizes | Parameters |
|---------------|--------|---------------|------------|------------|
| **Single-Scale** | 13 models | ~9K samples each | (66×94) to (206×294) | ~1–2M each |
| **Multi-Scale** | 1 model | ~117K samples mixed | Variable dimensions | ~1–2M total |

**Grid dimensions by parcel size** (parcel + 100% padding at 3m resolution):

| Parcel | Grid Size | Total Cells |
|--------|-----------|-------------|
| 27m | 66×94 | 6,204 |
| 33m | 78×110 | 8,580 |
| 39m | 90×126 | 11,340 |
| 45m | 102×142 | 14,484 |
| 51m | 110×158 | 17,380 |
| 57m | 122×174 | 21,228 |
| 63m | 134×190 | 25,460 |
| 69m | 146×214 | 31,244 |
| 75m | 158×230 | 36,340 |
| 81m | 170×246 | 41,820 |
| 87m | 182×262 | 47,684 |
| 93m | 194×278 | 53,932 |
| 99m | 206×294 | 60,564 |

#### Multi-Scale Architecture

**Key innovations**:

1. **Size Embedding**: MLP encodes (H, W) → 16D → broadcast to spatial dimensions
2. **Adaptive Pooling**: Bottleneck fixed at 8×8 regardless of input size
3. **Dynamic Upsampling**: Decoder matches skip connection sizes automatically

**Input**: (B, 3, H, W) — terrain, buildings, landuse (H, W varies)  
**Output**: (B, 6, H, W) — uq, vq, uz, vz, Ex, Hx (resized to input)

#### Training Configuration

```yaml
parcel_sizes: [27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99]
training_samples: ~9,000 per size (3 SAIL replicates)
total_samples: ~117,000 (multi-scale) or ~9,000 (single-scale)
max_epochs: 200
batch_size:
  single_scale_small: 32   # Parcels ≤69m
  single_scale_large: 16   # Parcels >69m (memory)
  multi_scale: 16           # Mixed sizes
early_stopping: patience=20
optimizer: Adam(lr=0.001)
lr_scheduler: ReduceLROnPlateau(patience=10, factor=0.5)
data_split: 85% train / 15% validation
```

### Key Hypotheses

1. **Accuracy Trade-off**: Single-scale models will achieve 1–3% higher R² due to specialization
2. **Training Efficiency**: Multi-scale training time ≈ 2–3× single model, but 80% savings vs training all 13
3. **Deployment Benefits**: Multi-scale offers 92% storage reduction (1 vs 13 models) + full size coverage

### Resource Requirements

**Single-Scale Jobs** (13 jobs via array 0–12):
- Time: 6 hours, Memory: 32GB, GPU: 1× A100

**Multi-Scale Job** (1 job):
- Time: 12 hours, Memory: 128GB, GPU: 1× A100

### Expected Results

| Model Type | Expected R² | Training Time | Storage |
|------------|-------------|---------------|---------|
| Single-scale (each) | 0.997 | ~2–6h | ~50MB |
| **Single Total (×13)** | **0.997** | **~50h** | **~650MB** |
| **Multi-Scale** | **0.99–0.995** | **~12h** | **~50MB** |

### Decision Criteria

- **R² (multi) > 0.99 across all sizes** → Deploy multi-scale
- **R² (multi) 0.98–0.99 on most sizes** → Use multi-scale (acceptable)
- **R² (multi) < 0.98 on many sizes** → Keep single-scale
- **Size-dependent performance** → Hybrid approach

### Running the Experiment

```bash
# Full pipeline (HPC)
cd hpc/exp7_multiscale_unet
bash run_exp7.sh all

# Or phase by phase
bash run_exp7.sh single   # Train 13 single-scale models (array job 0-12)
bash run_exp7.sh multi    # Train 1 multi-scale model
bash run_exp7.sh analyze  # Generate comparison plots

# Check status
bash run_exp7.sh status
```

### Analysis Outputs

```bash
python experiments/exp7_multiscale_unet/analyze_results.py \
    --results-file results/exp7_multiscale_unet/training_results.json \
    --output-dir results/exp7_multiscale_unet/analysis
```

**Generated files**: `comparison_table.csv`, `r2_comparison.png`, `per_field_accuracy_51m.png`, `training_time.png`, `summary_report.txt`

### Current Status

*Experiment designed. Configured for all 13 parcel sizes (27–99m). Ready for HPC execution.*

---

## Experiment 8: Performance Benchmarking

**Documentation**: [experiments/exp8_performance_benchmark/README.md](experiments/exp8_performance_benchmark/README.md)  
**Report**: [experiments/exp8_performance_benchmark/PERFORMANCE_REPORT.md](experiments/exp8_performance_benchmark/PERFORMANCE_REPORT.md)  
**Results**: [results/exp8_performance_benchmark/](results/exp8_performance_benchmark/)

### Research Question

What are the performance bottlenecks preventing "coffee break" (<15 min) QD optimization? How much speedup can we achieve through code optimization?

### Motivation

Initial Experiment 6 runs revealed unexpectedly slow performance:

| Model | Time for 1K gens | Projected 10K gens | Solutions/gen |
|-------|------------------|--------------------|--------------:|
| SVGP  | 572.5s (9.5 min) | ~96 min | 1024 |
| U-Net | 1486.1s (24.8 min) | ~248 min | 1024 |
| Hybrid| 1588.5s (26.5 min) | ~265 min | 1024 |

**Paradox**: SVGP (GP model) is 2.6× faster than U-Net (neural network). This indicates Python overhead dominates, not model inference.

### Identified Bottlenecks

#### 1. Feature Computation (MAJOR - all models)
- Creates new `ParametricEncoding` per solution (1024×/gen)
- Sequential Python loops in `express()` and `calculate_planning_features()`
- `scipy.ndimage.label()` called per solution
- **Estimated**: ~400ms/gen (70% of SVGP time)

#### 2. Domain Grid Construction (MAJOR - U-Net/Hybrid)
- Duplicates encoding work (already done for features)
- Sequential Python loops creating 66×94 grids
- **Estimated**: ~800ms/gen (54% of U-Net time)

#### 3. Cold Air Flux Computation (MODERATE - U-Net/Hybrid)
- Python for-loop with per-sample ROI masking
- **Estimated**: ~100ms/gen (7% of U-Net time)

#### 4. Model Inference (ACCEPTABLE)
- SVGP: ~50-80ms/gen (well-optimized)
- U-Net: ~150ms/gen (acceptable)

### Proposed Optimizations

| ID | Optimization | Target | Expected Speedup |
|----|--------------|--------|------------------|
| O1 | Reuse ParametricEncoding | Features | 2-4× |
| O2 | Share heightmaps | Features + Domain | 2× |
| O3 | Vectorize domain construction | Domain | 10-20× |
| O4 | Vectorize flux computation | Flux | 5-10× |
| O5 | Numba-JIT features | Features | 3-5× |

### Expected Results After Optimization

| Model | Current | Optimized | Speedup |
|-------|---------|-----------|---------|
| SVGP  | 572ms/gen | ~180ms/gen | 3.2× |
| U-Net | 1486ms/gen | ~320ms/gen | 4.6× |
| Hybrid| 1588ms/gen | ~380ms/gen | 4.2× |

**Projected 10K generations:**
- SVGP: 96 min → ~30 min
- U-Net: 248 min → ~53 min
- Hybrid: 265 min → ~63 min

**Alternative - reduce generations:**
- 5K gens + optimization: SVGP ~15 min ✓

### Benchmarks

The experiment runs micro-benchmarks for each component:

1. **Feature Computation** - Baseline vs optimized (reused encoding)
2. **Domain Construction** - Baseline vs vectorized NumPy
3. **Flux Computation** - Python loop vs vectorized
4. **Model Inference** - SVGP and U-Net across batch sizes
5. **Full Pipeline** - End-to-end comparison

### Running the Experiment

```bash
# Submit to HPC
bash hpc/exp8_performance_benchmark/run_exp8.sh submit

# Run locally
bash hpc/exp8_performance_benchmark/run_exp8.sh local

# View results
bash hpc/exp8_performance_benchmark/run_exp8.sh results
```

### Current Status

*Experiment created. Ready for execution.*

---

## Experiment 9: KLAM_21 Boundary Analysis

**Documentation**: [experiments/exp9_klam_boundary_analysis/README.md](experiments/exp9_klam_boundary_analysis/README.md)  
**Results**: [results/exp9_klam_boundary_analysis/](results/exp9_klam_boundary_analysis/)

### Research Question

How do different landuse configurations affect cold air flow patterns, particularly at domain boundaries? Is the harsh velocity transition at the landuse 7→2 boundary a physical artifact or realistic behavior?

### Motivation

KLAM_21 simulations show a sharp velocity gradient where landuse changes from 7 (vegetation/free space) to 2 (low-density buildings) at the parcel boundary. This experiment isolates whether the transition is caused by landuse changes, terrain slope, or their combination — and whether it distorts the optimization objective.

### Experimental Design

A **street canyon** test layout (two parallel buildings with open channel, 51m parcel) maximizes flow visibility. **Eight configurations** systematically vary landuse and terrain:

| Config | Name | Landuse | Terrain |
|--------|------|---------|---------|
| 1 | Current (baseline) | 7 → 2 at parcel edge | 1° slope upwind, flat at parcel |
| 2 | Uniform vegetation | 7 everywhere | 1° slope upwind, flat at parcel |
| 3 | Design-only built | 7 everywhere, buildings = 2 | 1° slope upwind, flat at parcel |
| 4 | Gradual transition | 7 → 4 → 2 (30m zone) | 1° slope upwind, flat at parcel |
| 5 | Flat terrain | 7 → 2 at parcel edge | Flat everywhere |
| 6 | Boundary after parcel | 7 (upwind+parcel) → 2 (downwind) | 1° slope upwind, flat at parcel |
| 7 | Continuous slope | 7 → 2 at parcel edge | 1° slope everywhere |
| 8 | Boundary after + continuous | 7 (upwind+parcel) → 2 (downwind) | 1° slope everywhere |

### Key Hypotheses

1. **Landuse 7→2 boundary creates sharp velocity transition** (Config 1 vs 2)
2. **Design-only landuse (Config 3) eliminates domain-wide boundary artifacts**
3. **Gradual landuse transition (Config 4) smooths the flow gradient**
4. **Terrain slope (Config 5 vs 1) contributes independently to cold air production**

### Quantitative Metrics (ROI: parcel + downwind)

- Mean/max velocity at 2m, mean cold air content (Ex), mean/total cold air flux
- Velocity and Ex standard deviation (flow uniformity)

### Running the Experiment

```bash
# Runs locally (~3-5 minutes, no HPC needed)
python experiments/exp9_klam_boundary_analysis/compare_klam_configurations.py \
    --output-dir results/exp9_klam_boundary_analysis \
    --parcel-size 51 \
    --xy-scale 3.0
```

### Analysis Outputs

**Generated files** in `results/exp9_klam_boundary_analysis/`:
- `comparison_plots/` — velocity fields, cold air content, flux, landuse, velocity profiles, metrics bar charts
- `metrics_summary.csv` — quantitative comparison across all 8 configs

### Current Status

*Experiment designed. Runs locally in ~3–5 minutes. Ready for execution.*

---

## Summary of Key Findings

### 1. Physics Insights

| Finding | Implication |
|---------|-------------|
| GRZ dominates cold air flux (ρ = -0.948) | Focus on site coverage in design |
| Height has negligible effect at 2m | Height limits for other reasons (views, shadows) |
| Orientation is negligible | Bar layout orientation doesn't matter |

### 2. Machine Learning Insights

| Finding | Implication |
|---------|-------------|
| Optimized-trained GP generalizes best | Use SAIL archives for training |
| Random-trained fails on optimized data | Don't use random sampling for GP training |
| 2000 inducing points optimal | Computational cost acceptable |
| K-means init provides marginal benefit | Use it (low cost, small gain) |

### 3. Recommended Configuration

```yaml
# GP Surrogate Configuration
model: SVGP
num_inducing: 2000
init_method: kmeans
kernel: Matern25_ARD_62D
training_data: sail_archives  # Not random!

# Training
max_epochs: 200
early_stopping_patience: 20
lr_warmup_epochs: 10
batch_size: 1024
learning_rate: 0.01
```

### 4. Performance Summary

#### SVGP Model (Production Configuration: 2500 Inducing Points)

| Metric | Value | Config |
|--------|-------|--------|
| **Best R² (in-domain)** | **0.946** | Optimized data, 5000 ind. pts |
| **Practical R² (in-domain)** | **0.936** | Optimized data, 2500 ind. pts |
| **Cross-domain R² (opt→combined)** | **0.810** | Optimized training |
| **Cross-domain R² (rand→opt)** | **-0.133** | Random training (fails!) |
| **Spearman ρ** | **~0.97** | Ranking fidelity |
| **95% CI Coverage** | **96.8%** | Well-calibrated |
| **RMSE** | **1.37** | Practical config |
| **MAE** | **0.99** | Practical config |
| **Prediction Speed** | **< 5ms** | Fast inference |
| **Training Time** | **~31 min** | 2500 ind. pts |

#### U-Net Model (Small Generic)

| Metric | Value |
|--------|-------|
| **Overall R²** | **0.997** |
| **Best Field R² (uq)** | **0.9998** |
| **Lowest Field R² (vq)** | **0.992** |
| **Overall MSE** | **0.0027** |
| **Overall MAE** | **0.021** |
| **Prediction Speed** | **~2ms** |
| **Throughput** | **~465 samples/sec** |
| **Training Time** | **~1.3 hours** |

#### Model Comparison

| Model | Best R² | Prediction Speed | Training Data | Output | Key Advantage |
|-------|---------|------------------|---------------|---------|---------------|
| **SVGP** | 0.946 | <5ms | ~26k samples | Scalar + uncertainty | Uncertainty quantification |
| **U-Net** | **0.997** | ~2ms | ~27k samples | **Spatial fields** | Highest accuracy + visualization |

**Conclusion**: Both surrogates achieve excellent accuracy. U-Net is superior in accuracy (99.7% vs 94.6% variance explained) and provides full spatial field predictions, making it ideal for both optimization and detailed visualization. SVGP provides valuable uncertainty estimates for exploration-exploitation balance.

---

## Running the Experiments

### Experiment 1: GP Training Data Comparison

```bash
# Full pipeline (HPC)
bash hpc/exp1_gp_training_data/run_gp_experiment.sh all

# Or phase by phase
bash hpc/exp1_gp_training_data/run_gp_experiment.sh 1   # Data generation
bash hpc/exp1_gp_training_data/run_gp_experiment.sh 2   # Dataset preparation
bash hpc/exp1_gp_training_data/run_gp_experiment.sh 3   # GP training
bash hpc/exp1_gp_training_data/run_gp_experiment.sh 4   # Evaluation
```

### Experiment 2: Flux Sensitivity

```bash
cd experiments/exp2_flux_sensitivity/flux_sensitivity
python run_experiment.py --output-dir ../../../results/flux_sensitivity
python analyze_results.py --input-dir ../../../results/flux_sensitivity
```

### Experiment 3: Hyperparameter Optimization

```bash
# Submit HPO jobs
sbatch hpc/exp3_hpo/submit_gp_hpo.sh

# Analyze results
python experiments/analyze_hpo_results.py \
    --results-dir results/hyperparameterization \
    --output-dir results/hyperparameterization/analysis

# Cross-domain evaluation
sbatch hpc/exp3_hpo/submit_evaluate_hpo_cross_domain.sh
```

### Experiment 4: MAP-Elites with Offline GP

```bash
# Run parameter sweep
sbatch hpc/exp4_mapelites_gp/submit_mapelites_gp_sweep.sh

# Validate top solutions with real KLAM_21
sbatch hpc/exp4_mapelites_gp/submit_validate_archive.sh

# Analyze results
python experiments/exp4_mapelites_gp/analyze_mapelites_gp.py \
    --results-dir results/mapelites_gp \
    --output-dir results/mapelites_gp/analysis
```

### Experiment 5: U-Net Surrogate

```bash
# Full training (submit all 12 configs)
bash hpc/exp5_unet/submit_all_unet_configs.sh

# Or single configuration
sbatch --export=ALL,DATA_TYPE=random,LOSS_TYPE=mse_grad,SEED=42 hpc/exp5_unet/submit_unet_training.sh

# Analyze results
python experiments/exp5_unet/analyze_unet_results.py \
    --results-dir results/exp5_unet/unet_experiment \
    --output-dir results/exp5_unet/unet_experiment/analysis
```

### Experiment 6a: Model Comparison (SVGP vs U-Net)

```bash
python experiments/exp6_model_comparison/compare_svgp_unet.py \
    --svgp-model results/exp3_hpo/best_model.pth \
    --unet-model results/exp5_unet/unet_experiment/best_model.pth \
    --output-dir results/exp6_model_comparison
```

### Experiment 6b: QD Comparison (MAP-Elites with U-Net)

```bash
# Submit all 24 jobs (8 configs × 3 seeds)
bash hpc/exp6_qd_comparison/submit_all_qd_configs.sh

# Analyze results
python experiments/exp6_qd_comparison/analyze_unet_optimization.py \
    --results-dir results/exp6_qd_comparison \
    --output-dir results/exp6_qd_comparison/analysis
```

### Experiment 7: Multi-Scale U-Net Comparison

```bash
cd hpc/exp7_multiscale_unet
bash run_exp7.sh all        # Full pipeline
bash run_exp7.sh status     # Check progress
```

### Experiment 8: Performance Benchmarking

```bash
bash hpc/exp8_performance_benchmark/run_exp8.sh all
bash hpc/exp8_performance_benchmark/run_exp8.sh results
```

### Experiment 9: KLAM_21 Boundary Analysis

```bash
# Runs locally (~3-5 minutes)
python experiments/exp9_klam_boundary_analysis/compare_klam_configurations.py \
    --output-dir results/exp9_klam_boundary_analysis \
    --parcel-size 51 --xy-scale 3.0
```
