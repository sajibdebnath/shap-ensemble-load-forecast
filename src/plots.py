"""
Visualisation utilities for the physics-informed ensemble paper.

Reproduces the key figures from:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Figures produced
----------------
Fig. 3 (a) : Piecewise parabolic temperature–demand scatter + envelope
Fig. 3 (b) : Monthly demand box-plot by year
Fig. 4     : Observed vs. predicted scatter + residual density (extreme subset)
Fig. 5     : Full test-year demand time series with extreme-event zoom
Fig. 6     : Physics-informed vs. unconstrained ensemble (Dec 2024 cold snap)
Fig. 7     : Global SHAP feature importance bar chart
Fig. 8     : Regime-comparison SHAP attribution (normal vs. extreme)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style settings ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "observed": "#1f77b4",
    "ensemble": "#d62728",
    "cnn": "#ff7f0e",
    "transformer": "#2ca02c",
    "lstm": "#9467bd",
    "extreme": "#e377c2",
    "envelope": "#ffd700",
    "fit": "#d62728",
    "normal": "#1f77b4",
    "shap_blue": "#2196F3",
    "shap_red": "#F44336",
}

OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ── Fig. 3(a) – Temperature–demand scatter ────────────────────────────────────

def plot_parabolic_envelope(
    temperature: np.ndarray,
    demand: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """
    Plot piecewise parabolic temperature–demand relationship with ±2σ band.

    Parameters
    ----------
    temperature : np.ndarray, shape (N,) — air temperature in °C
    demand : np.ndarray, shape (N,) — ERCOT system load in GW
    save : bool — save figure to results/figures/fig3a_parabola.png

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.losses.physics_loss import ercot_parabolic_envelope, T0

    T_grid = np.linspace(temperature.min(), temperature.max(), 300)
    D_fit = ercot_parabolic_envelope(T_grid).numpy()
    sigma = 3_000.0  # approximate 1σ of residuals in MW

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(
        temperature, demand / 1e3,
        c=COLORS["observed"], alpha=0.25, s=4, label="Hourly observations (2018–2022)",
    )
    ax.fill_between(
        T_grid, (D_fit - 2 * sigma) / 1e3, (D_fit + 2 * sigma) / 1e3,
        color=COLORS["envelope"], alpha=0.45, label=r"±2σ(T) envelope",
    )
    ax.plot(
        T_grid, D_fit / 1e3,
        color=COLORS["fit"], lw=2, label="Piecewise parabolic fit D(T)",
    )
    ax.axvline(T0, color="grey", ls="--", lw=1, label=f"T₀ = {T0}°C")
    ax.text(T0 + 0.3, demand.max() / 1e3 * 0.95, "cooling\nregime", fontsize=8, color="grey")
    ax.text(T0 - 7, demand.max() / 1e3 * 0.95, "heating\nregime", fontsize=8, color="grey")

    ax.set_xlabel("Air temperature (°C)")
    ax.set_ylabel("ERCOT demand (GW)")
    ax.set_title("(a) Piecewise parabolic temperature–demand relationship")
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    if save:
        fig.savefig(OUTDIR / "fig3a_parabola.png")
        print(f"Saved → {OUTDIR / 'fig3a_parabola.png'}")
    return fig


# ── Fig. 4 – Observed vs. predicted scatter + residual density ────────────────

def plot_scatter_and_residuals(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    extreme_mask: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """
    Observed-vs-predicted scatter (top row) and residual density (bottom).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) — in GW
    predictions : dict {model_label: np.ndarray}
    extreme_mask : np.ndarray of bool, shape (N,)
    save : bool
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 9))

    for col, (label, y_pred) in enumerate(predictions.items()):
        ax = axes[0, col]
        y_t = y_true / 1e3
        y_p = y_pred / 1e3

        # All points
        ax.scatter(y_t[~extreme_mask], y_p[~extreme_mask],
                   c=COLORS["observed"], s=4, alpha=0.3, label="Normal")
        # Extreme events
        ax.scatter(y_t[extreme_mask], y_p[extreme_mask],
                   c=COLORS["extreme"], s=8, alpha=0.7, label="Extreme (Hampel)")

        # 1:1 reference
        lim = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lim, lim, "k--", lw=1, label="1:1 reference")

        # OLS fit
        coeffs = np.polyfit(y_t, y_p, 1)
        ax.plot(lim, np.polyval(coeffs, lim), "-", color=COLORS["fit"], lw=1.5,
                label=f"slope={coeffs[0]:.3f}")

        ax.set_xlabel("Observed demand (GW)")
        ax.set_ylabel("Predicted demand (GW)")
        ax.set_title(f"({chr(97 + col)}) {label}")
        ax.legend(fontsize=8)

    # ── Residual density (bottom row, merged) ────────────────────────────────
    ax_res = axes[1, n_models // 2]
    axes[1, 0].set_visible(False)
    if n_models > 2:
        axes[1, -1].set_visible(False)

    for label, y_pred in predictions.items():
        residuals = (y_pred[extreme_mask] - y_true[extreme_mask]) / 1e3
        ax_res.hist(residuals, bins=60, density=True, alpha=0.5, label=label)

    ax_res.axvline(0, color="k", lw=1)
    ax_res.set_xlabel("Residual ŷ − y (GW)  —  extreme-event subset")
    ax_res.set_ylabel("Density")
    ax_res.set_title("(b) Residual distribution on Hampel-flagged extreme events")
    ax_res.legend()

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig4_scatter_residuals.png")
        print(f"Saved → {OUTDIR / 'fig4_scatter_residuals.png'}")
    return fig


# ── Fig. 5 – Full test-year time series ──────────────────────────────────────

def plot_test_timeseries(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred_ensemble: np.ndarray,
    extreme_mask: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """
    Full 2024 test-year observed vs. ensemble forecast time series.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex — hourly UTC timestamps for test set
    y_true : np.ndarray, shape (N,) — in GW
    y_pred_ensemble : np.ndarray, shape (N,)
    extreme_mask : np.ndarray of bool, shape (N,)
    save : bool
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    y_t = y_true / 1e3
    y_p = y_pred_ensemble / 1e3

    # ── Full year ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(timestamps, y_t, color=COLORS["observed"], lw=0.7, label="Observed")
    ax.plot(timestamps, y_p, color=COLORS["ensemble"], lw=0.7, alpha=0.8,
            label="PI Ensemble forecast")
    ax.scatter(timestamps[extreme_mask], y_t[extreme_mask],
               c=COLORS["extreme"], s=6, zorder=5, label="Hampel extreme events")
    ax.set_title("(a) Full 2024 test window — observed vs. physics-informed ensemble forecast")
    ax.set_ylabel("ERCOT demand (GW)")
    ax.legend(loc="upper right")

    # ── August 2024 heat dome zoom ────────────────────────────────────────────
    ax = axes[1]
    mask_aug = (timestamps >= "2024-07-27") & (timestamps <= "2024-08-24")
    ax.plot(timestamps[mask_aug], y_t[mask_aug], color=COLORS["observed"], lw=1, label="Observed")
    ax.plot(timestamps[mask_aug], y_p[mask_aug], color=COLORS["ensemble"], lw=1,
            label="PI Ensemble forecast")
    ax.set_title("(b) 30-day zoom — August 2024 heat dome")
    ax.set_ylabel("ERCOT demand (GW)")
    ax.legend()

    # ── January 2024 Arctic outbreak zoom ────────────────────────────────────
    ax = axes[2]
    mask_jan = (timestamps >= "2024-01-08") & (timestamps <= "2024-01-22")
    ax.plot(timestamps[mask_jan], y_t[mask_jan], color=COLORS["observed"], lw=1, label="Observed")
    ax.plot(timestamps[mask_jan], y_p[mask_jan], color=COLORS["ensemble"], lw=1,
            label="PI Ensemble forecast")
    extreme_jan = extreme_mask[mask_jan]
    ax.scatter(timestamps[mask_jan][extreme_jan], y_t[mask_jan][extreme_jan],
               c=COLORS["extreme"], s=30, zorder=5, label="Hampel extreme events")
    ax.set_title("(c) 14-day zoom — January 2024 Arctic outbreak")
    ax.set_ylabel("ERCOT demand (GW)")
    ax.legend()

    plt.tight_layout()
    if save:
        fig.savefig(OUTDIR / "fig5_timeseries.png")
        print(f"Saved → {OUTDIR / 'fig5_timeseries.png'}")
    return fig


# ── Fig. 7 – Global SHAP feature importance ───────────────────────────────────

def plot_global_shap_importance(
    importance: Dict[str, float],
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of global SHAP feature importance.

    Parameters
    ----------
    importance : dict {feature_name: mean_abs_shap}  — sorted descending
    save : bool
    """
    features = list(importance.keys())[::-1]
    values = list(importance.values())[::-1]

    # Colour by feature family
    family_colors = {
        "lagged_load": "#1f77b4",
        "air_temp": "#ff7f0e",
        "feels_like": "#ff7f0e",
        "humidity": "#2ca02c",
        "wind_speed": "#2ca02c",
        "precip": "#2ca02c",
        "w_data": "#2ca02c",
        "hour": "#9467bd",
        "day_of_week": "#9467bd",
        "month": "#9467bd",
        "weekend_flag": "#9467bd",
        "holiday_flag": "#9467bd",
    }
    colors = [family_colors.get(f, "grey") for f in features]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, values, color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005 * max(values), bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8)

    patches = [
        mpatches.Patch(color="#1f77b4", label="Load (autoregressive)"),
        mpatches.Patch(color="#ff7f0e", label="Thermal"),
        mpatches.Patch(color="#2ca02c", label="Meteorological"),
        mpatches.Patch(color="#9467bd", label="Calendar"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    ax.set_xlabel("Mean |φ| (MW) — global SHAP importance")
    ax.set_title("Global SHAP feature importance — physics-informed ensemble on ERCOT 2024–2025 test set")
    plt.tight_layout()

    if save:
        fig.savefig(OUTDIR / "fig7_global_shap.png")
        print(f"Saved → {OUTDIR / 'fig7_global_shap.png'}")
    return fig


# ── Fig. 8 – Regime comparison SHAP ─────────────────────────────────────────

def plot_regime_shap_comparison(
    normal_importance: Dict[str, float],
    extreme_importance: Dict[str, float],
    save: bool = True,
) -> plt.Figure:
    """
    Side-by-side SHAP comparison: normal days vs. Hampel-flagged extremes.

    Parameters
    ----------
    normal_importance  : dict {feature_name: mean_abs_shap}
    extreme_importance : dict {feature_name: mean_abs_shap}
    save : bool
    """
    features = list(normal_importance.keys())
    normal_vals = np.array([normal_importance[f] for f in features])
    extreme_vals = np.array([extreme_importance[f] for f in features])

    # Sort by extreme importance
    order = np.argsort(extreme_vals)[::-1]
    features = [features[i] for i in order]
    normal_vals = normal_vals[order]
    extreme_vals = extreme_vals[order]

    y_pos = np.arange(len(features))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_pos + bar_height / 2, normal_vals, bar_height,
            color=COLORS["normal"], alpha=0.85, label="Normal days")
    ax.barh(y_pos - bar_height / 2, extreme_vals, bar_height,
            color=COLORS["extreme"], alpha=0.85, label="Hampel-flagged extreme events")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Mean |φ| (MW) — SHAP feature importance by regime")
    ax.set_title("Regime-dependent SHAP attribution — normal vs. extreme-event feature importance")
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(OUTDIR / "fig8_regime_shap.png")
        print(f"Saved → {OUTDIR / 'fig8_regime_shap.png'}")
    return fig
