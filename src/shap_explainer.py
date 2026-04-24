"""
SHAP DeepExplainer-Based Interpretability Framework.

Implements the explainability layer from:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

SHAP ensemble attribution (Eq. 12):
    φ_ens_(j,i) = w*_CNN · φ_CNN_(j,i) + w*_T · φ_T_(j,i)

Global feature importance (Eq. 13):
    I_j = (1/N_test) Σ_i |φ_ens_(j,i)|

Computed separately for:
  (i)  all test samples
  (ii) Hampel-flagged extreme events
  (iii) normal days

Feature names (in order, matching input tensor columns):
    0: lagged_load  — lag-1 hourly demand (MW)
    1: air_temp     — air temperature (°C)
    2: feels_like   — apparent temperature (°C)
    3: humidity     — relative humidity (%)
    4: wind_speed   — surface wind speed (m/s)
    5: precip       — hourly precipitation (mm)
    6: w_data       — encoded weather type
    7: hour         — hour-of-day  (0–23)
    8: day_of_week  — day-of-week  (1–7)
    9: month        — month        (1–12)
   10: weekend_flag — binary weekend indicator
   11: holiday_flag — binary U.S. federal-holiday indicator
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(
        "[SHAP] shap package not found. Install with: pip install shap==0.44.*"
    )

FEATURE_NAMES: List[str] = [
    "lagged_load",
    "air_temp",
    "feels_like",
    "humidity",
    "wind_speed",
    "precip",
    "w_data",
    "hour",
    "day_of_week",
    "month",
    "weekend_flag",
    "holiday_flag",
]


# ── Hampel filter for extreme-event flagging ─────────────────────────────────

def hampel_filter(
    series: np.ndarray,
    window: int = 720,   # 30 days × 24 hours
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Flag anomalous load observations using the Hampel filter.

    A point is flagged if |y_i − median(window)| > n_sigma × MAD(window),
    where MAD = median(|y_i − median|).

    Parameters
    ----------
    series : np.ndarray
        Hourly demand time series, shape (N,).
    window : int
        Rolling window size in hours (default: 720 = 30 days).
    n_sigma : float
        Detection threshold in units of MAD (default: 3.0).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,) — True for flagged extreme events.
    """
    n = len(series)
    flags = np.zeros(n, dtype=bool)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = series[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad > 0 and np.abs(series[i] - med) > n_sigma * mad:
            flags[i] = True

    return flags


# ── SHAP DeepExplainer wrapper ───────────────────────────────────────────────

class EnsembleSHAPExplainer:
    """
    SHAP attribution for the CNN–Transformer physics-informed ensemble.

    Computes branch-level SHAP values via DeepExplainer and combines them
    using the ensemble weights (Eq. 12).

    Parameters
    ----------
    cnn_model : tf.keras.Model
        Trained CNN branch (outputs [embedding, forecast]).
    transformer_model : tf.keras.Model
        Trained Transformer branch (outputs [embedding, forecast]).
    w_cnn : float
        Ensemble weight for the CNN branch.
    w_transformer : float
        Ensemble weight for the Transformer branch.
    background_size : int
        Number of background samples for SHAP (default: 500).
    feature_names : list of str, optional
        Feature name list for attribution display.
    """

    def __init__(
        self,
        cnn_model: tf.keras.Model,
        transformer_model: tf.keras.Model,
        w_cnn: float = 0.38,
        w_transformer: float = 0.62,
        background_size: int = 500,
        feature_names: Optional[List[str]] = None,
    ):
        if not SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap==0.44.*")

        self.cnn_model = cnn_model
        self.transformer_model = transformer_model
        self.w_cnn = w_cnn
        self.w_transformer = w_transformer
        self.background_size = background_size
        self.feature_names = feature_names or FEATURE_NAMES

        self._cnn_explainer: Optional[shap.DeepExplainer] = None
        self._transformer_explainer: Optional[shap.DeepExplainer] = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_forecast_model(self, branch_model: tf.keras.Model) -> tf.keras.Model:
        """
        Extract a forecast-only sub-model (second output) for SHAP.
        DeepExplainer requires a single-output model.
        """
        inputs = branch_model.input
        forecast_output = branch_model.outputs[1]   # ŷ branch
        return tf.keras.Model(inputs=inputs, outputs=forecast_output)

    def _stratified_background(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Draw a stratified background sample from training data.
        Stratification is by temperature quartile × season to ensure
        representation of both summer and winter regimes.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, T, F)
        y_train : np.ndarray, shape (N,)   — demand values for stratification
        n_samples : int

        Returns
        -------
        np.ndarray
            Background set, shape (n_samples, T, F).
        """
        # Use demand quartile as a simple stratifier (correlates with season)
        quartiles = np.percentile(y_train, [25, 50, 75])
        idx_q = np.digitize(y_train, quartiles)
        bg_indices = []
        per_stratum = n_samples // 4

        for q in range(4):
            stratum_idx = np.where(idx_q == q)[0]
            chosen = np.random.choice(
                stratum_idx,
                size=min(per_stratum, len(stratum_idx)),
                replace=False,
            )
            bg_indices.extend(chosen.tolist())

        bg_indices = np.array(bg_indices[:n_samples])
        return X_train[bg_indices]

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int = 42,
    ) -> "EnsembleSHAPExplainer":
        """
        Initialise DeepExplainer for both branches using a stratified
        background drawn from training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, T=24, F=12)
        y_train : np.ndarray, shape (N,)
        seed : int

        Returns
        -------
        self
        """
        np.random.seed(seed)

        background = self._stratified_background(
            X_train, y_train, self.background_size
        )

        cnn_forecast_model = self._make_forecast_model(self.cnn_model)
        transformer_forecast_model = self._make_forecast_model(self.transformer_model)

        self._cnn_explainer = shap.DeepExplainer(
            cnn_forecast_model, background
        )
        self._transformer_explainer = shap.DeepExplainer(
            transformer_forecast_model, background
        )

        print(
            f"[SHAP] Explainers initialised. "
            f"Background size: {background.shape[0]} samples."
        )
        return self

    # ── Computing attributions ────────────────────────────────────────────────

    def compute_shap_values(
        self, X_test: np.ndarray
    ) -> np.ndarray:
        """
        Compute ensemble SHAP values for a test set (Eq. 12).

        φ_ens_(j,i) = w*_CNN · φ_CNN_(j,i) + w*_T · φ_T_(j,i)

        Parameters
        ----------
        X_test : np.ndarray, shape (N, T=24, F=12)

        Returns
        -------
        np.ndarray
            Ensemble SHAP values, shape (N, T, F).
        """
        if self._cnn_explainer is None or self._transformer_explainer is None:
            raise RuntimeError("Call .fit() before .compute_shap_values().")

        phi_cnn = self._cnn_explainer.shap_values(X_test)          # (N, T, F)
        phi_transformer = self._transformer_explainer.shap_values(X_test)  # (N, T, F)

        phi_cnn = np.array(phi_cnn)
        phi_transformer = np.array(phi_transformer)

        # Handle list output from some SHAP versions
        if isinstance(phi_cnn, list):
            phi_cnn = phi_cnn[0]
        if isinstance(phi_transformer, list):
            phi_transformer = phi_transformer[0]

        phi_ensemble = self.w_cnn * phi_cnn + self.w_transformer * phi_transformer
        return phi_ensemble  # (N, T, F)

    # ── Global importance ─────────────────────────────────────────────────────

    def global_importance(
        self,
        phi: np.ndarray,
        regime_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute global feature importance I_j (Eq. 13).

        I_j = (1/N) Σ_i |φ_ens_(j,i)|

        Importance is aggregated over the time dimension (mean) before
        computing the mean absolute value over samples.

        Parameters
        ----------
        phi : np.ndarray, shape (N, T, F)
            Ensemble SHAP values.
        regime_mask : np.ndarray of bool, shape (N,), optional
            If provided, restrict computation to masked samples.

        Returns
        -------
        dict
            {feature_name: mean_absolute_shap} sorted descending by importance.
        """
        if regime_mask is not None:
            phi = phi[regime_mask]

        # Average SHAP values over the time dimension → (N, F)
        phi_mean_time = np.mean(phi, axis=1)

        # Mean absolute SHAP per feature → (F,)
        importance = np.mean(np.abs(phi_mean_time), axis=0)

        result = dict(zip(self.feature_names, importance.tolist()))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    # ── Regime comparison ─────────────────────────────────────────────────────

    def regime_comparison(
        self,
        phi: np.ndarray,
        extreme_mask: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compare SHAP importance between normal and extreme-event regimes.

        Parameters
        ----------
        phi : np.ndarray, shape (N, T, F)
            Ensemble SHAP values for all test samples.
        extreme_mask : np.ndarray of bool, shape (N,)
            True for Hampel-flagged extreme events.

        Returns
        -------
        Tuple[dict, dict]
            (normal_importance, extreme_importance) — both dicts map
            feature_name → mean |SHAP| value.
        """
        normal_mask = ~extreme_mask
        normal_imp = self.global_importance(phi, regime_mask=normal_mask)
        extreme_imp = self.global_importance(phi, regime_mask=extreme_mask)
        return normal_imp, extreme_imp

    # ── Rank-stability via Kendall's τ ────────────────────────────────────────

    def bootstrap_rank_stability(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstraps: int = 20,
        seed: int = 0,
    ) -> float:
        """
        Assess SHAP rank-stability via Kendall's τ across background resamples.

        Parameters
        ----------
        X_train : np.ndarray, shape (N_train, T, F)
        y_train : np.ndarray, shape (N_train,)
        X_test  : np.ndarray, shape (N_test, T, F)
        n_bootstraps : int
        seed : int

        Returns
        -------
        float
            Mean Kendall's τ across all bootstrap pairs.
        """
        from scipy.stats import kendalltau

        rankings = []
        for b in range(n_bootstraps):
            np.random.seed(seed + b)
            self.fit(X_train, y_train, seed=seed + b)
            phi_b = self.compute_shap_values(X_test)
            imp_b = self.global_importance(phi_b)
            rankings.append(list(imp_b.keys()))  # ordered feature names

        # Compute mean Kendall's τ over all pairs
        taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                rank_i = {f: k for k, f in enumerate(rankings[i])}
                rank_j = {f: k for k, f in enumerate(rankings[j])}
                r_i = [rank_i[f] for f in self.feature_names]
                r_j = [rank_j[f] for f in self.feature_names]
                tau, _ = kendalltau(r_i, r_j)
                taus.append(tau)

        mean_tau = float(np.mean(taus))
        print(f"[SHAP] Kendall τ rank-stability: {mean_tau:.3f} "
              f"(n_bootstraps={n_bootstraps})")
        return mean_tau
