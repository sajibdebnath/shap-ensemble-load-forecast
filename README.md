# Interpretable Physics-Informed Load Forecasting for U.S. Grid Resilience

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.14](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://tensorflow.org/)
[![SHAP 0.44](https://img.shields.io/badge/SHAP-0.44-green.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SHAP-Guided Ensemble Validation in Hybrid Deep Learning Under Extreme Weather**

> Md Abubakkar¹, Sajib Debnath², Md. Uzzal Mia³  
> ¹ Dept. of Computer Science, Midwestern State University, Dallas, TX  
> ² O&M Analytics, AES Clean Energy, The AES Corporation, Louisville, CO  
> ³ Dept. of Information and Communication Engineering, Pabna University of Science and Technology

---

## Overview

This repository contains the official implementation of the physics-informed interpretable ensemble framework for U.S. grid load forecasting presented in the above paper.

Accurate short-term electricity load forecasting is critical for grid reliability, but prevailing deep learning models remain opaque — limiting operator trust during extreme weather events. We address this gap through a unified framework that jointly delivers:

1. **Physics-informed learning** — a composite training loss derived from ERCOT's piecewise parabolic temperature–demand relationship, penalising predictions that violate the empirical thermal-response envelope or that imply implausible hour-over-hour ramps.
2. **Deep ensemble fusion** — a dual-stream CNN–Transformer architecture whose branch predictions are combined via validation-optimised convex weights.
3. **SHAP interpretability** — DeepExplainer-based attribution identifying *why* the model predicts a given load level, with explicit regime comparison between normal and Hampel-flagged extreme-weather events.

---

## Key Results (ERCOT 2024–2025 Test Window)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Accuracy (%) |
|-------|----------|-----------|----------|--------------|
| Linear Regression (baseline) | 7,390 | 7,538 | 11.10 | 88.90 |
| Weather-Informed LSTM | 1,402 | 1,490 | 2.20 | 97.80 |
| Weather-Informed Transformer | 892 | 992 | 1.41 | 98.59 |
| Attention CNN–LSTM | 1,431 | 1,915 | 2.53 | 97.47 |
| **Proposed — CNN branch (PI)** | 1,103 | 1,285 | 1.72 | 98.28 |
| **Proposed — Transformer branch (PI)** | 803 | 918 | 1.28 | 98.72 |
| **Proposed — PI Ensemble** | **713** | **812** | **1.18** | **98.82** |

**Extreme-event improvement:** The ensemble reduces MAPE by **20.7%** relative to its standalone Transformer branch and by **40.5%** relative to its standalone CNN branch on 142 Hampel-flagged extreme events (January 2024 Arctic outbreak, August 2024 heat dome, December 2024 cold snap).

**Physics constraint ablation:** The full physics-informed loss reduces extreme-event RMSE by **14.7%** over the unconstrained baseline.

**SHAP regime shift:** Wind speed rises from rank 7 (normal days) to rank 3 (extreme events); precipitation rises from rank 8 to rank 4 — exposing a regime shift that is invisible to black-box forecasters.

---

## Key Contributions

1. **Physics-Informed Loss** — Differentiable penalty terms derived from the ERCOT piecewise parabolic temperature–demand envelope, incorporating parabolic (steady-state) and ramp (transient) constraints.

2. **Dual-Stream Hybrid Architecture** — 1-D CNN branch for local multivariate motif extraction + Transformer branch for long-range temporal dependencies, fused via simplex-projected OLS ensemble weights.

3. **SHAP DeepExplainer Attribution** — Ensemble-level SHAP values combining branch attributions by ensemble weights; global importance rankings and normal-vs-extreme regime comparison with Kendall τ rank stability (τ = 0.91).

4. **Empirical Demonstration** — Eight years of ERCOT + ASOS data (2018–2025), covering the January 2024 Arctic outbreak, August 2024 heat dome, and December 2024 cold snap.

---

## Repository Structure

```
pi-load-forecast/
├── train.py                     # Main training script
├── evaluate.py                  # Post-training evaluation + SHAP attribution
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── models/
│   │   ├── cnn_branch.py        # 1-D CNN branch (local feature extraction)
│   │   └── transformer_branch.py# Transformer branch (long-range dependencies)
│   ├── losses/
│   │   └── physics_loss.py      # Composite physics-informed loss (MSE + parabolic + ramp)
│   ├── ensemble/
│   │   └── weighted_ensemble.py # Validation-optimised convex ensemble fusion
│   ├── explainability/
│   │   └── shap_explainer.py    # SHAP DeepExplainer + Hampel filter + regime comparison
│   ├── preprocessing/
│   │   └── data_pipeline.py     # Data loading, feature engineering, normalisation
│   ├── evaluation/
│   │   └── metrics.py           # MAE, RMSE, MAPE, extreme-event evaluation
│   └── visualization/
│       └── plots.py             # All paper figures (Figs. 3–8)
│
├── data/
│   ├── raw/                     # Place downloaded CSVs here (see instructions below)
│   ├── processed/               # Auto-generated by preprocessing pipeline
│   └── instructions/
│       └── DATA_DOWNLOAD.md     # Step-by-step data download guide
│
├── notebooks/                   # Jupyter notebooks (exploratory analysis)
│
├── results/
│   ├── figures/                 # Generated plots (PNG, 300 DPI)
│   ├── tables/                  # Evaluation result CSVs
│   └── checkpoints/             # Saved model weights + scaler
│
├── docs/
│   └── methodology.md           # Extended methodology notes
│
└── tests/                       # Unit tests
    └── test_losses.py
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU recommended (paper uses RTX A6000 48 GB); CPU-only training is supported but slow

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/pi-load-forecast.git
cd pi-load-forecast

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

The model is trained on **eight years** of public data:

| Source | Variable | Range | Records |
|--------|----------|-------|---------|
| [ERCOT Load Archive](https://www.ercot.com/gridinfo/load/load_hist) | Hourly system demand | 29,360–85,435 MW | 70,128 h |
| [ASOS / IEM](https://mesonet.agron.iastate.edu/request/download.phtml?network=TX_ASOS#) | Air temp, RH, wind speed, precip | 3 Texas stations | Hourly |

**Temporal split:**
- Train: 2018–2022 (43,824 h)
- Validation: 2023 (8,760 h)
- Test: 2024–2025 (17,544 h)

### Download Instructions

See [`data/instructions/DATA_DOWNLOAD.md`](data/instructions/DATA_DOWNLOAD.md) for step-by-step guidance on downloading and placing ERCOT and ASOS files.

**Quick automated ASOS download:**
```bash
python - <<'EOF'
import requests, pathlib

STATIONS = ["BKS", "JDD", "TME"]
for sid in STATIONS:
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={sid}&data=tmpf&data=relh&data=sknt&data=p01i&data=wxcodes"
        f"&year1=2018&month1=1&day1=1&year2=2025&month2=12&day2=31"
        f"&tz=UTC&format=comma&latlon=no&missing=M&trace=T&direct=no&report_type=3"
    )
    out = pathlib.Path(f"data/raw/asos_{sid}.csv")
    out.write_text(requests.get(url, timeout=300).text)
    print(f"Saved {out}")
EOF
```

---

## Usage

### 1. Train the Model

```bash
python train.py \
  --ercot_path  data/raw/ercot_load_2018_2025.csv \
  --asos_paths  data/raw/asos_BKS.csv data/raw/asos_JDD.csv data/raw/asos_TME.csv \
  --lambda1 0.1 \
  --lambda2 0.05 \
  --batch_size 64 \
  --max_epochs 100 \
  --run_name pi_ensemble
```

Training produces checkpoints in `results/checkpoints/`.

### 2. Evaluate and Run SHAP Attribution

```bash
python evaluate.py \
  --ercot_path  data/raw/ercot_load_2018_2025.csv \
  --asos_paths  data/raw/asos_BKS.csv data/raw/asos_JDD.csv data/raw/asos_TME.csv \
  --checkpoint_dir results/checkpoints \
  --run_name pi_ensemble \
  --shap_background 500 \
  --shap_n_bootstraps 20
```

To skip SHAP (faster, evaluation metrics only):
```bash
python evaluate.py ... --skip_shap
```

### 3. Ablation Study (Physics Constraint)

Reproduce Table IV by running with different lambda values:

```bash
# No physics constraints
python train.py ... --lambda1 0.0 --lambda2 0.0 --run_name ablation_no_physics

# Parabolic only
python train.py ... --lambda1 0.1 --lambda2 0.0 --run_name ablation_parabolic_only

# Ramp only
python train.py ... --lambda1 0.0 --lambda2 0.05 --run_name ablation_ramp_only

# Full PI loss (paper default)
python train.py ... --lambda1 0.1 --lambda2 0.05 --run_name pi_ensemble
```

### 4. Reproducing Figures

All figures from the paper are generated automatically by `evaluate.py` and saved under `results/figures/`:

| File | Paper Figure |
|------|-------------|
| `fig3a_parabola.png` | Fig. 3(a) — Piecewise parabolic envelope |
| `fig4_scatter_residuals.png` | Fig. 4 — Scatter + residual density |
| `fig5_timeseries.png` | Fig. 5 — Test-year time series + zoom |
| `fig7_global_shap.png` | Fig. 7 — Global SHAP importance |
| `fig8_regime_shap.png` | Fig. 8 — Regime comparison SHAP |

---

## Model Architecture

### Physics-Informed Composite Loss

```
L_b = L_MSE + λ₁ · L_parabolic + λ₂ · L_ramp

L_parabolic = (1/N) Σ max(0, |ŷᵢ − D(Tᵢ)| − ε(Tᵢ))²
L_ramp      = (1/N) Σ max(0, |ŷᵢ − ŷᵢ₋₁| − Δ_max)²

ERCOT envelope:  D(T) = 47.2T² − 1560.6T + 51230  (T < 18.5°C)
                        52.4T² − 864.5T  + 35523.9 (T ≥ 18.5°C)
Hyperparameters: λ₁ = 0.1, λ₂ = 0.05, Δ_max = 4800 MW/h
```

### Ensemble Weights

Weights are obtained by projecting the unconstrained OLS solution onto the unit simplex:

```
(w*_CNN, w*_T) = argmin_{w ∈ Δ₁} ‖y_val − (w_CNN·ŷ_CNN + w_T·ŷ_T)‖²
```

Paper result: **(w*_CNN, w*_T) = (0.38, 0.62)**

### SHAP Ensemble Attribution

```
φ_ens_(j,i) = w*_CNN · φ_CNN_(j,i) + w*_T · φ_T_(j,i)
I_j         = (1/N) Σᵢ |φ_ens_(j,i)|
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{abubakkar2026pi_load,
  author    = {Abubakkar, Md and Debnath, Sajib and Mia, Md. Uzzal},
  title     = {Interpretable Physics-Informed Load Forecasting for {U.S.} Grid Resilience:
               {SHAP}-Guided Ensemble Validation in Hybrid Deep Learning Under Extreme Weather},
  booktitle = {Proceedings of the IEEE},
  year      = {2026},
}
```

**Related prior work:**
```bibtex
@inproceedings{debnath2025naps,
  author  = {Debnath, Sajib and others},
  title   = {Extreme Weather Grid Load Forecasting Using Weather-Informed {LSTM} and
             {Transformer} Machine Learning Models},
  booktitle = {Proc. 57th North Amer. Power Symp. (NAPS)},
  year    = {2025},
  doi     = {10.1109/naps66256.2025.11272315},
}

@article{debnath2026ieeeaccess,
  author  = {Debnath, Sajib and others},
  title   = {Hybrid Multi-Scale Deep Learning Enhanced Electricity Load Forecasting
             Using Attention-Based Convolutional Neural Network and {LSTM} Model},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {13423--13444},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3656545},
}
```

---

## Acknowledgements

The authors thank ERCOT and the NOAA/IEM ASOS network for providing publicly available datasets, and AES Clean Energy for operational grid-reliability context.

---

## License

This project is released under the [MIT License](LICENSE).
