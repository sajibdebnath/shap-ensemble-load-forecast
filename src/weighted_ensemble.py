"""
Validation-Optimised Weighted Ensemble Fusion.

Implements the ensemble combination strategy from:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Ensemble forecast (Eq. 10):
    ŷ_ens = w_CNN · ŷ_CNN + w_T · ŷ_T

Optimal weights (Eq. 11) are found by projecting the unconstrained OLS
solution onto the unit simplex (w_CNN + w_T = 1, w_CNN, w_T ≥ 0).

Reported validation-set result: (w*_CNN, w*_T) = (0.38, 0.62).
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple


# ── Simplex projection helper ────────────────────────────────────────────────

def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the unit simplex {w : Σw = 1, w ≥ 0}.

    Uses the O(n log n) algorithm from Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray
        Input weight vector, shape (n,).

    Returns
    -------
    np.ndarray
        Projected weight vector on the unit simplex, shape (n,).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(n) + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ── Closed-form OLS weight estimation ────────────────────────────────────────

def fit_ensemble_weights(
    y_val: np.ndarray,
    y_hat_cnn: np.ndarray,
    y_hat_transformer: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate validation-optimised ensemble weights via constrained OLS.

    Solves:
        (w*_CNN, w*_T) = argmin_{w ∈ Δ₁} ‖y_val − (w_CNN·ŷ_CNN + w_T·ŷ_T)‖²

    where Δ₁ is the unit simplex.

    Parameters
    ----------
    y_val : np.ndarray
        Validation-set ground-truth demand values, shape (N,).
    y_hat_cnn : np.ndarray
        CNN branch predictions on the validation set, shape (N,).
    y_hat_transformer : np.ndarray
        Transformer branch predictions on the validation set, shape (N,).

    Returns
    -------
    Tuple[float, float]
        (w_cnn, w_transformer) — convex combination weights summing to 1.

    Notes
    -----
    The unconstrained OLS solution in the 2-model case reduces to a
    single scalar regression; the simplex projection clips it to [0, 1].
    """
    # Stack branch predictions as columns: X shape (N, 2)
    X = np.column_stack([y_hat_cnn, y_hat_transformer])

    # Unconstrained OLS: w = (X^T X)^{-1} X^T y
    w_ols, *_ = np.linalg.lstsq(X, y_val, rcond=None)

    # Project onto simplex to enforce w ≥ 0 and Σw = 1
    w_simplex = project_onto_simplex(w_ols)

    w_cnn = float(w_simplex[0])
    w_transformer = float(w_simplex[1])

    return w_cnn, w_transformer


# ── Ensemble predictor ───────────────────────────────────────────────────────

class WeightedEnsemble:
    """
    Validation-optimised weighted ensemble of CNN and Transformer branches.

    Parameters
    ----------
    w_cnn : float
        Weight assigned to the CNN branch (default: 0.38 from paper).
    w_transformer : float
        Weight assigned to the Transformer branch (default: 0.62 from paper).

    Attributes
    ----------
    w_cnn_ : float
    w_transformer_ : float
    is_fitted_ : bool
    """

    # Default weights reported in the paper (ERCOT 2023 validation set)
    DEFAULT_W_CNN = 0.38
    DEFAULT_W_TRANSFORMER = 0.62

    def __init__(
        self,
        w_cnn: float = DEFAULT_W_CNN,
        w_transformer: float = DEFAULT_W_TRANSFORMER,
    ):
        assert abs(w_cnn + w_transformer - 1.0) < 1e-6, (
            "Ensemble weights must sum to 1."
        )
        self.w_cnn_ = w_cnn
        self.w_transformer_ = w_transformer
        self.is_fitted_ = True

    @classmethod
    def from_validation(
        cls,
        y_val: np.ndarray,
        y_hat_cnn: np.ndarray,
        y_hat_transformer: np.ndarray,
    ) -> "WeightedEnsemble":
        """
        Construct a WeightedEnsemble by fitting weights on validation data.

        Parameters
        ----------
        y_val : np.ndarray
            Validation ground-truth demand, shape (N,).
        y_hat_cnn : np.ndarray
            CNN branch validation predictions, shape (N,).
        y_hat_transformer : np.ndarray
            Transformer branch validation predictions, shape (N,).

        Returns
        -------
        WeightedEnsemble
            Instance with fitted weights.
        """
        w_cnn, w_transformer = fit_ensemble_weights(
            y_val, y_hat_cnn, y_hat_transformer
        )
        print(
            f"[Ensemble] Fitted weights — "
            f"w_CNN={w_cnn:.3f}, w_Transformer={w_transformer:.3f}"
        )
        return cls(w_cnn=w_cnn, w_transformer=w_transformer)

    def predict(
        self,
        y_hat_cnn: np.ndarray,
        y_hat_transformer: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the weighted ensemble prediction (Eq. 10).

        ŷ_ens = w_CNN · ŷ_CNN + w_T · ŷ_T

        Parameters
        ----------
        y_hat_cnn : np.ndarray
            CNN branch predictions, shape (N,).
        y_hat_transformer : np.ndarray
            Transformer branch predictions, shape (N,).

        Returns
        -------
        np.ndarray
            Ensemble predictions, shape (N,).
        """
        return self.w_cnn_ * y_hat_cnn + self.w_transformer_ * y_hat_transformer

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble("
            f"w_cnn={self.w_cnn_:.3f}, "
            f"w_transformer={self.w_transformer_:.3f})"
        )
