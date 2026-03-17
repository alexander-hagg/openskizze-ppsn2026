# OpenSKIZZE KLAM-21 Optimization

This repository implements **Surrogate-Assisted Quality-Diversity** optimization for urban building layout design, using physics-based **KLAM_21** cold airflow simulation.

**License**: AGPLv3. For commercial licensing, contact info@haggdesign.de.

## Overview

This project develops **surrogate models** that replace expensive KLAM_21 cold airflow simulations (~5-10 min each) with fast predictions (<5ms), enabling **Quality-Diversity (QD) optimization** for urban building layouts. Two surrogate approaches are provided:

1. **SVGP (Sparse Variational Gaussian Process)** - Fast scalar predictions with uncertainty estimates
2. **U-Net (Convolutional Neural Network)** - High-accuracy predictions with full spatial field outputs

### Surrogate Model Performance

| Model | R² | Prediction Speed | VRAM (batch=1024) | Output | Key Advantage |
|-------|----|-----------------|--------------------|--------|---------------|
| **SVGP** | 0.946 | <5ms | ~0.5 GB | Scalar + uncertainty | Uncertainty quantification |
| **U-Net** | **0.997** | ~2ms | **~9 GB** | Spatial fields (6 channels) | Highest accuracy + visualization |

**VRAM Requirements**:
- **SVGP**: ~0.5 GB (CPU or GPU compatible)
- **U-Net**: ~9 GB for batch size 1024 (requires GPU, recommend NVIDIA A100 40GB)

### Key Experimental Findings

| Finding | Source |
|---------|--------|
| **GRZ (site coverage) dominates cold air flux** (ρ = -0.948) | [Flux Sensitivity](reports/flux_sensitivity/) |
| **Train on SAIL archives, not random data** (avg R² 0.806 vs 0.328) | [GP Training Comparison](EXPERIMENTS.md#experiment-1) |
| **2000 inducing points optimal** for SVGP (R² = 0.946) | [HPO Report](reports/hpo/) |
| **U-Net achieves 99.7% accuracy** on KLAM fields | [U-Net Results](EXPERIMENTS.md#experiment-5) |

📖 **See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment descriptions and complete results.**

---

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate openskizze_klam_qd

# Install package
pip install -e .

# Verify dependencies
python check_dependencies.py
```

### Run QD Optimization

**Option 1: SAIL (online surrogate training)**
```bash
# SAIL with KLAM-21 physics (slow but trains surrogate online)
python run_sail.py --evaluation klam --replicate 1

# SAIL with fast flood-fill proxy
python run_sail.py --evaluation flood_fill --replicate 1
```

**Option 2: MAP-Elites with offline surrogates (fast, "coffee break" optimization)**
```bash
# Pure MAP-Elites with pre-trained SVGP (~30 min for 10K generations)
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model svgp --parcel-size 27 --generations 10000

# Pure MAP-Elites with U-Net (~53 min for 10K generations, highest accuracy)
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model unet --parcel-size 27 --generations 10000

# Hybrid: U-Net fitness + SVGP uncertainty for exploration
python experiments/exp6_qd_comparison/run_mapelites_offline.py \
    --model hybrid --ucb-lambda 1.0 --parcel-size 27 --generations 10000
```

---

## How It Works: QD Optimization with Surrogate Models

### Quality-Diversity (QD) Optimization

**QD algorithms** like MAP-Elites discover diverse, high-performing solutions by:
1. Maintaining an **archive** of solutions across behavioral dimensions (features)
2. Generating new candidate solutions via mutation/crossover
3. Evaluating fitness and features
4. Adding solutions to archive if they improve their behavioral niche

**Result**: Archive containing thousands of diverse, high-quality building layouts.

### Surrogate-Assisted QD

Physics simulations (KLAM_21) are too slow for QD (~5-10 min each × 100K evaluations = 580 hours). **Surrogates replace physics with fast predictions**:

**Approach 1: Online Training (SAIL)**
- Start with surrogate trained on small initial dataset
- Use surrogate to propose promising candidates
- Evaluate best candidates with real physics
- Retrain surrogate with new data
- Repeat for N generations

**Approach 2: Offline Surrogate (MAP-Elites)**
- Train surrogate on pre-collected SAIL archive data (~26K samples)
- Run pure MAP-Elites using only surrogate predictions
- No physics evaluations during optimization ("coffee break" speed)
- Optionally validate final archive with real physics

**Approach 3: Hybrid (U-Net + SVGP)**
- Use U-Net for accurate fitness predictions (R² = 0.997)
- Use SVGP uncertainty estimates for exploration bonus
- Combines best accuracy with exploration-exploitation balance
- Fitness = `U-Net(x) + λ × SVGP_std(x)` where λ controls exploration

### Optimization Times

| Method | Time (10K generations) | Accuracy | Use Case |
|--------|------------------------|----------|----------|
| Pure KLAM_21 | ~580 hours | Perfect | Ground truth only |
| SAIL (online GP) | ~96 hours | R²=0.946 | Research/validation |
| MAP-Elites + SVGP | **~30 min** | R²=0.946 | Fast iteration |
| MAP-Elites + U-Net | **~53 min** | R²=0.997 | Production ready |
| MAP-Elites + Hybrid | **~63 min** | R²=0.997 + exploration | Best quality |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | This file - overview and quickstart |
| **[TECHNICAL.md](TECHNICAL.md)** | Architecture, data formats, KLAM_21 interface |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | All experiments, results, and detailed findings |
| **[reports/](reports/)** | LaTeX reports with full analysis |

---

## Project Structure

```
openskizze-klam21-optimization/
├── run_sail.py                         # SAIL (online surrogate training)
├── experiments/
│   ├── exp6_qd_comparison/
│   │   └── run_mapelites_offline.py   # MAP-Elites with offline surrogates
│   ├── exp1_gp_training_data/          # GP training data comparison
│   ├── exp2_flux_sensitivity/          # Morphology sensitivity analysis
│   ├── exp3_hpo/                       # SVGP hyperparameter optimization
│   ├── exp5_unet/                      # U-Net surrogate training
│   ├── exp7_multiscale_unet/           # Multi-scale U-Net comparison
│   └── models/                         # Model architectures (unet.py, svgp.py)
├── hpc/                                # HPC batch scripts
├── reports/                            # LaTeX reports (PDF)
├── domain_description/                 # KLAM_21 evaluation interface
├── encodings/parametric/               # Building genome encoding (60D)
└── optimization/                       # QD algorithms (SAIL, MAP-Elites)
```

---

## References

- **SAIL**: Gaier et al. (2018). [Data-Efficient Design Exploration through Surrogate-Assisted Illumination](https://arxiv.org/abs/1806.05865)
- **SVGP**: Hensman et al. (2015). [Scalable Variational Gaussian Process Classification](https://arxiv.org/abs/1411.2005)
- **KLAM_21**: DWD (German Weather Service) cold airflow model
- **PyRibs**: [pyribs.org](https://pyribs.org/) - Quality-Diversity optimization library
- **GPyTorch**: [gpytorch.ai](https://gpytorch.ai/) - Gaussian Process library
