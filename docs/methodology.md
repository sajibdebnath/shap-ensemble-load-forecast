# Methodology Reference

This document provides extended notes on the methodology, hyperparameters, and design choices in the physics-informed interpretable ensemble framework.

---

## 1. Input Representation

Each training sample is a 24-hour window **X ∈ ℝ^{24×12}** with the following 12 features:

| Index | Feature | Units | Type |
|-------|---------|-------|------|
| 0 | Lagged load L(d−1) | MW | Autoregressive |
| 1 | Air temperature | °C | Meteorological |
| 2 | Feels-like temperature | °C | Meteorological |
| 3 | Relative humidity | % | Meteorological |
| 4 | Wind speed | m/s | Meteorological |
| 5 | Precipitation | mm/h | Meteorological |
| 6 | W-data (weather type) | categorical | Meteorological |
| 7 | Hour of day | 0–23 | Calendar |
| 8 | Day of week | 1–7 | Calendar |
| 9 | Month | 1–12 | Calendar |
| 10 | Weekend flag | 0/1 | Calendar |
| 11 | Holiday flag | 0/1 | Calendar |

Continuous features are standardised to zero mean and unit variance (fitted on training set only). The forecast target is the **next-hour demand** y_{t+1}.

---

## 2. ERCOT Piecewise Parabolic Envelope

The temperature–demand relationship is modelled as a piecewise parabola (calibrated in [7]):

```
D(T) = 47.2T² − 1560.6T + 51230     (T < 18.5°C,  heating regime)
       52.4T² − 864.5T  + 35523.9   (T ≥ 18.5°C,  cooling regime)
```

The inflection point **T₀ = 18.5°C** separates the heating and cooling regimes. The tolerance band is ε(T) = 2σ(T), corresponding to the two-sided prediction interval of the fit (σ ≈ 3,000 MW from training residuals).

---

## 3. Physics-Informed Loss

The composite loss for each branch b ∈ {CNN, Transformer}:

```
L_b = L_MSE + λ₁ · L_parabolic + λ₂ · L_ramp
```

**Parabolic constraint (Eq. 7):**
```
L_parabolic = (1/N) Σᵢ max(0, |ŷᵢ − D(Tᵢ)| − ε(Tᵢ))²
```
Penalises predictions outside the ±2σ envelope around D(T).

**Ramp constraint (Eq. 9):**
```
L_ramp = (1/N) Σᵢ max(0, |ŷᵢ − ŷᵢ₋₁| − Δ_max)²
```
Penalises hour-over-hour changes exceeding Δ_max = 4,800 MW/h (99.5th percentile of training first differences).

**Hyperparameters (tuned on 2023 validation set):**
- λ₁ = 0.1 (parabolic weight)
- λ₂ = 0.05 (ramp weight)

---

## 4. CNN Branch

Two stacked 1-D convolutional blocks:
```
Input (24 × 12) → Conv1D(k=3, c=64) → BN → ReLU → MaxPool(s=2)
               → Conv1D(k=3, c=64) → BN → ReLU → MaxPool(s=2)
               → GlobalAvgPool → Dense(64) → z_CNN ∈ ℝ⁶⁴
               → Dense(1) → ŷ_CNN
```
Dropout = 0.2 after each block.

---

## 5. Transformer Branch

```
Input (24 × 12) → Linear Projection (d=64) → Positional Encoding
               → Encoder Block × 2:
                   [MultiHead Self-Attention (h=4, d_k=16)
                    + LayerNorm + FFN(d_ff=256) + LayerNorm]
               → GlobalAvgPool → Dense(64) → z_T ∈ ℝ⁶⁴
               → Dense(1) → ŷ_T
```

Positional encoding uses the standard sinusoidal scheme from Vaswani et al. (2017).

---

## 6. Ensemble Fusion

The ensemble forecast is a convex combination of branch predictions:

```
ŷ_ens = w_CNN · ŷ_CNN + w_T · ŷ_T     (w_CNN + w_T = 1, both ≥ 0)
```

Weights are fitted by projecting the unconstrained OLS solution onto the unit simplex using the validation set (2023). Reported result: **(w_CNN, w_T) = (0.38, 0.62)**.

---

## 7. Hampel Filter for Extreme Events

The Hampel filter identifies anomalous load points in the test set:

```
flag(i) = True if |y_i − median(window)| > 3 × MAD(window)
```

- **Window:** 720 hours (30-day rolling)
- **Threshold:** 3 × MAD (Median Absolute Deviation)
- **Applied only to test set** (never during training)

---

## 8. SHAP DeepExplainer

- **Backend:** SHAP DeepExplainer (modified DeepLIFT propagation)
- **Background set:** 500 samples, stratified by season × temperature decile
- **Ensemble attribution:** φ_ens = w_CNN · φ_CNN + w_T · φ_T (Eq. 12)
- **Global importance:** I_j = (1/N) Σᵢ |φ_ens_(j,i)| (Eq. 13)
- **Rank stability:** Kendall τ = 0.91 across 20 bootstrap resamples

---

## 9. Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr=1e-3, β₁=0.9, β₂=0.999) |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping patience | 10 (on validation MAE) |
| Dropout | 0.2 (both branches) |
| Hardware | NVIDIA RTX A6000 (48 GB), Ubuntu 22.04 |
| Seeds | 3 random seeds; reported values are means |
| Seed-to-seed MAPE std | < 0.12% |
