# WP1.2: SVGP vs Small U-Net Comparison

**Goal**: Determine which offline surrogate model to use for MAP-Elites optimization in the GUI.

**Date**: December 5, 2025  
**Status**: Ready to run  
**Est. Time**: 30-60 minutes (HPC with GPU)

---

## Overview

This experiment compares two surrogate models:

1. **SVGP (Sparse Variational Gaussian Process)**
   - Trained on SAIL archives (HPO optimized)
   - 2500 inducing points
   - Input: 62D (60D genome + parcel width + height)
   - R² = 0.93 (from HPO experiments)

2. **Small U-Net (Generic)**
   - Trained on 1500 KLAM_21 samples
   - Input: 66×94 grid (terrain, buildings, landuse)
   - R² = 0.99 vs KLAM_21
   - Throughput: 1235 samples/s (from benchmark)

---

## Comparison Metrics

### 1. Agreement
- **Pearson r**: Linear correlation between predictions
- **Spearman ρ**: Rank correlation (most important for QD)

### 2. Ranking Fidelity
- **Top-100 overlap**: How many of the top 100 solutions agree between models

### 3. Throughput
- **Samples/second**: Evaluation speed on GPU

### 4. Calibration
- **RMSE**: Absolute prediction error
- **Distribution match**: Are fitness distributions similar?

---

## Decision Criteria

From FINAL_WEEK_IMPLEMENTATION_PLAN.md:

- **If ρ > 0.85 AND U-Net throughput > 10× SVGP** → Use U-Net
- **If ρ > 0.90 AND similar throughput** → Use SVGP (more mature)
- **Else** → Use both (ensemble) or based on integration ease

---

## Usage

### Quick Local Test (100 samples)

```bash
cd /home/alex/Documents/_cloud/Funded_Projects/OpenSKIZZE/code/openskizze-klam21-optimization
bash experiments/exp6_model_comparison/test_comparison.sh
```

Review results in `results/model_comparison_test/`

### Full HPC Run (1000 samples)

```bash
# Submit job
sbatch hpc/exp6_model_comparison/submit_comparison.sh

# Check status
squeue -u $USER

# View results
cat logs/model_comparison_*.out
```

---

## Output Files

All saved to `results/model_comparison/`:

1. **svgp_vs_unet_comparison.json**
   - All metrics
   - Recommendation
   - Throughput comparison
   - Summary statistics

2. **comparison_plots.png**
   - Scatter plot (agreement)
   - Residual plot
   - Distribution comparison
   - Ranking comparison
   - Bland-Altman plot
   - Metrics summary

3. **predictions.npz**
   - Raw predictions from both models
   - Genomes, widths, heights
   - For further analysis

---

## Expected Results

Based on preliminary analysis:

- **Spearman ρ**: 0.85-0.95 (high agreement expected)
- **U-Net throughput**: ~1000-1200 samples/s
- **SVGP throughput**: ~200-500 samples/s
- **Throughput ratio**: 2-5× (U-Net faster)

**Likely recommendation**: Either model works, choose based on integration ease or use U-Net for speed.

---

## Next Steps

1. **Review results**:
   - Check `svgp_vs_unet_comparison.json` for recommendation
   - View `comparison_plots.png` for visual validation

2. **If recommendation = "Use U-Net"**:
   - Proceed to WP3.1: Create `evaluation_unet.py`
   - Integrate U-Net into GUI optimization

3. **If recommendation = "Use SVGP"**:
   - Adapt existing SVGP code for GUI
   - May need to create `evaluation_svgp.py`

4. **If recommendation = "Either" or "Both"**:
   - Implement U-Net first (faster, cleaner interface)
   - Optionally add SVGP as alternative

---

## Troubleshooting

### Model not found

```bash
# Check SVGP model exists
ls -lh results/hyperparameterization/model_optimized_ind2500_random_rep1.pth

# Check U-Net model exists
ls -lh results/unet_experiment/sail_mse_seed42/best_model.pth
```

### GPU issues

If CUDA out of memory:
- Reduce batch sizes: `--svgp-batch-size 128 --unet-batch-size 64`
- Or run on CPU: `--no-gpu` (much slower)

### Import errors

```bash
# Verify environment
conda activate openskizze_klam_qd
python -c "import torch; import gpytorch; print('OK')"
```

---

## Implementation Notes

### SVGP Evaluation
- Loads checkpoint with normalization parameters
- Evaluates on 62D input (genome + dimensions)
- Returns scalar fitness predictions

### U-Net Evaluation
- Converts genomes → phenotypes → spatial grids
- Predicts full KLAM_21 outputs (uq, vq, Ex, etc.)
- Computes fitness: Φ = mean(Ex) × mean(wind_speed)

### Comparison
- Evaluates identical set of 1000 diverse layouts (Sobol sequence)
- Measures agreement, ranking, and throughput
- Provides automated recommendation

---

## Contact

**Questions?** Check:
- FINAL_WEEK_IMPLEMENTATION_PLAN.md (WP1.2 section)
- REPOSITORY_ORGANIZATION_STRATEGY.md (migration plan)

**Issues?** Review:
- experiment log: `logs/model_comparison_*.out`
- error log: `logs/model_comparison_*.err`
