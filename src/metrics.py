"""
Evaluation Metrics and Reporting.

Implements standard time-series forecasting metrics (Eqs. 14–16) and
extreme-event subset evaluation via Hampel filter flagging.

Metrics reported in the paper (Tables II–III):
  MAE   : Mean Absolute Error (MW)
  RMSE  : Root Mean Squared Error (MW)
  MAPE  : Mean Absolute Percentage Error (%)
  Acc.  : Accuracy = 100 − MAPE (%)
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from src.explainability.shap_explainer import hampel_filter


# ── Core metric functions ─────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in MW."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in MW."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = 100 − MAPE (%)."""
    return 100.0 - mape(y_true, y_pred)


# ── Full evaluation report ────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model",
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute and print MAE, RMSE, MAPE, and Accuracy for a prediction set.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)
    label  : str
        Model or subset identifier for display.
    mask   : np.ndarray of bool, shape (N,), optional
        If provided, restrict evaluation to masked subset.

    Returns
    -------
    dict
        {'mae': float, 'rmse': float, 'mape': float, 'accuracy': float}
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }

    print(
        f"[{label}]  MAE={metrics['mae']:,.0f} MW | "
        f"RMSE={metrics['rmse']:,.0f} MW | "
        f"MAPE={metrics['mape']:.2f}% | "
        f"Acc={metrics['accuracy']:.2f}%"
    )
    return metrics


# ── Extreme-event evaluation ──────────────────────────────────────────────────

def evaluate_extreme_events(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    window: int = 720,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Evaluate all models on Hampel-flagged extreme-event subset.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth demand values.
    predictions : dict {model_name: np.ndarray}
        Dictionary mapping model labels to their predictions.
    window : int
        Hampel filter rolling window (default: 720 hours = 30 days).
    n_sigma : float
        Hampel filter threshold in MADs (default: 3.0).

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = [n_events, MAE, RMSE, MAPE, Accuracy].
    """
    extreme_mask = hampel_filter(y_true, window=window, n_sigma=n_sigma)
    n_extreme = extreme_mask.sum()
    print(f"\n[Extreme Events] Hampel-flagged events: {n_extreme} / {len(y_true)}")

    rows = []
    for label, y_pred in predictions.items():
        m = evaluate(y_true, y_pred, label=f"{label} [extreme]", mask=extreme_mask)
        rows.append({"model": label, "n_events": n_extreme, **m})

    df = pd.DataFrame(rows).set_index("model")
    return df


# ── Ablation study helper ─────────────────────────────────────────────────────

def ablation_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Format ablation study results as a DataFrame matching Table IV in the paper.

    Parameters
    ----------
    results : dict
        {config_name: {mae, rmse, mape, accuracy}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    baseline_rmse = None
    for cfg, metrics in results.items():
        row = {"Configuration": cfg, **metrics}
        rows.append(row)
        if baseline_rmse is None:
            baseline_rmse = metrics["rmse"]

    df = pd.DataFrame(rows).set_index("Configuration")
    if baseline_rmse:
        df["delta_rmse_pct"] = ((df["rmse"] - baseline_rmse) / baseline_rmse * 100).round(1)
    return df
