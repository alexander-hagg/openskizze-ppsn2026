# Experiment 9: KLAM_21 Boundary Analysis

## Overview

This experiment investigates the harsh transition observed at landuse boundaries in KLAM_21 simulations. The goal is to determine if the sharp velocity gradient at the landuse 7→2 boundary is a physical artifact or a realistic representation of cold air flow dynamics.

## Research Question

**How do different landuse configurations affect cold air flow patterns, particularly at domain boundaries?**

Specifically:
- Is the harsh transition from fast-flowing air (landuse 7) to slower flow (landuse 2) realistic?
- Can we isolate the flow effects around the building design from boundary artifacts?
- Do gradual landuse transitions or uniform vegetation improve flow realism?

## Experimental Design

### Street Canyon Test Design

A simple **street canyon** layout is used as the test case:
- **Two parallel buildings** on north and south edges of parcel
- **Open channel** between buildings (perpendicular to wind)
- **Parcel size**: 51m × 51m (17×17 cells)
- **Building dimensions**: 51m wide × 12m deep × 15m tall (5 floors)
- **Canyon width**: ~27m

This design maximizes flow channeling effects and makes velocity gradients highly visible.

### Five Configurations Tested

| Config | Name | Description | Landuse | Terrain |
|--------|------|-------------|---------|---------|
| **1** | Current | Standard setup | 7 (upwind) → 2 (parcel+downwind) | 1° slope upwind, flat at parcel |
| **2** | Uniform vegetation | All vegetation | 7 everywhere | 1° slope upwind, flat at parcel |
| **3** | Design-only built | Buildings in vegetation | 7 everywhere except under buildings (2) | 1° slope upwind, flat at parcel |
| **4** | Gradual transition | Smooth boundary | 7 → 4 → 2 (30m transition zone) | 1° slope upwind, flat at parcel |
| **5** | Flat terrain | No slope | 7 (upwind) → 2 (parcel+downwind) | 0° (flat everywhere) |
| **6** | Boundary after parcel | Boundary AFTER design | 7 (upwind+parcel) → 2 (downwind only) | 1° slope upwind, flat at parcel |
| **7** | Continuous slope | No terrain discontinuity | 7 (upwind) → 2 (parcel+downwind) | 1° slope everywhere (continuous) |
| **8** | Boundary after + continuous | Isolates building effects only | 7 (upwind+parcel) → 2 (downwind only) | 1° slope everywhere (continuous) |

### Configuration Details

**Config 1 (Current)**:
```
┌────────────────────────────────────────────┐
│  Landuse 7  │   Landuse 2 (parcel+right)  │
│  (upwind)   │                              │
│  1° slope ↗ │    Flat terrain              │
└────────────────────────────────────────────┘
```

**Config 2 (Uniform vegetation)**:
```
┌────────────────────────────────────────────┐
│        Landuse 7 everywhere                │
│  1° slope ↗ │    Flat terrain              │
└────────────────────────────────────────────┘
```

**Config 3 (Design-only built)**:
```
┌────────────────────────────────────────────┐
│  Landuse 7  │  Landuse 7, buildings = 2   │
│  (upwind)   │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │
│  1° slope ↗ │      (street canyon)         │
│             │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │
└────────────────────────────────────────────┘
```

**Config 4 (Gradual transition)**:
```
┌────────────────────────────────────────────┐
│ Landuse 7 │ 4 │   Landuse 2               │
│ (upwind)  │30m│  (parcel+right)           │
│ 1° slope ↗│   │   Flat terrain            │
└────────────────────────────────────────────┘
```

**Config 5 (Flat terrain)**:
```
┌────────────────────────────────────────────┐
│  Landuse 7  │   Landuse 2 (parcel+right)  │
│  (upwind)   │                              │
│    Flat (0°) terrain everywhere            │
└────────────────────────────────────────────┘
```

**Config 6 (Boundary after parcel)**:
```
┌────────────────────────────────────────────┐
│  Landuse 7       │  Parcel (7)  │  Landuse 2│
│  (upwind)        │              │ (downwind)│
│  1° slope ↗      │  Flat terrain│           │
└────────────────────────────────────────────┘
```
*This isolates the boundary effect to AFTER the design, not before.*

**Config 7 (Continuous slope)**:
```
┌────────────────────────────────────────────┐
│  Landuse 7  │   Landuse 2 (parcel+right)  │
│  (upwind)   │                              │
│  1° slope ↗ ↗ ↗ ↗ ↗ ↗ (continuous slope)  │
└────────────────────────────────────────────┘
```
*Eliminates terrain discontinuity at parcel edge - slope continues through entire domain.*

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Wind speed | 5.0 m/s |
| Wind direction | 270° (west) |
| Simulation duration | 14,400s (4 hours) |
| Grid resolution | 3m per cell |
| Domain extension | 100% upwind (left) |
| Output timestep | Final steady-state (t=14,400s) |

## Output Analysis

### Quantitative Metrics (ROI: parcel + downwind)

For each configuration, compute:
- **Mean velocity at 2m** (m/s)
- **Max velocity at 2m** (m/s)
- **Mean cold air content (Ex)** (100 J/m²)
- **Mean cold air height (Hx)** (m)
- **Mean cold air flux** (100 W/m²)
- **Total cold air flux** (100 W/m²)
- **Velocity standard deviation** (flow uniformity)
- **Ex standard deviation** (cold air distribution)

### Visualization Outputs

1. **Velocity fields** - Wind speed at 2m for all configs (side-by-side)
2. **Cold air content** - Ex fields showing cold air accumulation
3. **Cold air flux** - Ex × u_2m product (optimization objective)
4. **Landuse configurations** - Visual comparison of domain setups
5. **Velocity profiles** - Cross-sections along flow direction (centerline)
6. **Metrics bar charts** - Quantitative comparison across configs

## Expected Results

### Hypothesis 1: Landuse 7→2 boundary creates sharp transition
- **Config 1** (current) shows harsh velocity drop at parcel edge
- **Config 2** (uniform veg) shows smooth flow throughout
- **Difference** isolates boundary effect from building effects

### Hypothesis 2: Building-only landuse minimizes artifacts
- **Config 3** (design-only built) should show realistic flow perturbation
- Buildings create local velocity changes, but no domain-wide boundary
- Most "pure" representation of design impact

### Hypothesis 3: Gradual transition smooths flow
- **Config 4** (7→4→2) reduces sharp gradient
- May be more physically realistic for vegetated-to-urban transitions
- Trade-off: Adds complexity, may not reflect actual site conditions

### Hypothesis 4: Slope drives cold air, not just landuse
- **Config 5** (flat) isolates terrain effect
- Compare with **Config 1** to separate slope vs landuse contributions
- Expected: Reduced cold air production, lower velocities

## Running the Experiment

### Quick Test
```bash
cd /home/alex/Documents/_cloud/Funded_Projects/OpenSKIZZE/code/openskizze-klam21-optimization

python experiments/exp9_klam_boundary_analysis/compare_klam_configurations.py \
    --output-dir results/exp9_klam_boundary_analysis \
    --parcel-size 51 \
    --xy-scale 3.0
```

### Expected Runtime
- **Per configuration**: ~30-60 seconds (KLAM_21 simulation)
- **Total runtime**: ~3-5 minutes (5 configs + plotting)
- **Output size**: ~50-100 MB (grids + plots)

### Output Structure
```
results/exp9_klam_boundary_analysis/
├── config1_current/
│   └── examples/
│       ├── terrain.asc
│       ├── buildings.asc
│       ├── landuse.asc
│       ├── klam_21.in
│       └── results/
│           ├── uq014400.asc
│           ├── vq014400.asc
│           ├── Ex014400.asc
│           └── ...
├── config2_uniform_veg/
├── config3_design_only/
├── config4_gradual/
├── config5_flat/
├── comparison_plots/
│   ├── velocity_fields_comparison.png
│   ├── cold_air_content_comparison.png
│   ├── cold_air_flux_comparison.png
│   ├── landuse_comparison.png
│   ├── velocity_profiles.png
│   └── metrics_comparison.png
└── metrics_summary.csv
```

## Interpreting Results

### Key Questions to Answer

1. **How large is the velocity drop at the landuse boundary in Config 1?**
   - Look at velocity profile plot
   - Compare centerline velocity before/after parcel edge

2. **Does Config 3 (design-only) eliminate the harsh transition?**
   - Check if velocity field shows smooth approach to buildings
   - Compare metrics: is velocity more uniform?

3. **Does Config 4 (gradual) provide a middle ground?**
   - Is transition smoother than Config 1?
   - Are metrics closer to Config 2 or Config 1?

4. **How much does terrain slope contribute vs landuse?**
   - Compare Config 5 (flat) with Config 1
   - Check Ex and velocity differences

5. **Which configuration is most realistic for urban planning?**
   - Config 1: Current practice (may have artifacts)
   - Config 3: Cleanest isolation of building effects
   - Config 4: Most realistic urban-vegetation interface

### Decision Criteria

Based on results, recommend:
- **Keep Config 1**: If boundary effect is small or physically justified
- **Switch to Config 3**: If we want to isolate building design effects only
- **Switch to Config 4**: If gradual transitions are more realistic
- **Hybrid approach**: Use different configs for different analysis questions

## Physical Interpretation

### Landuse Categories in KLAM_21

| Code | Description | Surface Roughness | Cold Air Production |
|------|-------------|-------------------|---------------------|
| **7** | Free space / vegetation | High roughness | High (cooling surface) |
| **4** | Sparse vegetation | Medium roughness | Medium |
| **2** | Low-density buildings | Low roughness | Low (heat storage) |

### Expected Physical Behavior

**Landuse 7 (vegetation)**:
- High surface roughness → slower near-surface winds
- Radiative cooling → cold air production
- Katabatic flow source

**Landuse 2 (built)**:
- Lower roughness → potentially faster winds
- Heat storage → reduced cooling
- Katabatic flow transport

**Transition zone**:
- Abrupt change in surface properties
- May create unrealistic flow acceleration
- Real-world: Would have gradual transition

## Future Work

1. **Sensitivity to transition width** (Config 4)
   - Test 10m, 30m, 50m transition zones
   - Find optimal balance between realism and computation

2. **Alternative landuse configurations**
   - Test other KLAM landuse codes (3, 5, 6, 8, 9)
   - Match to actual site conditions

3. **Multiple building layouts**
   - Test courtyard, scattered, linear configurations
   - Check if boundary effect depends on design

4. **Real-world validation**
   - Compare with CFD simulations or measurements
   - Determine if KLAM_21 boundary behavior is realistic

## References

- KLAM_21 documentation (DWD)
- Flux Sensitivity Analysis (Experiment 2)
- TECHNICAL.md - KLAM_21 interface details
