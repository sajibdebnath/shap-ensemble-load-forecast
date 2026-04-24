"""
Physics-Informed Loss Functions.

Implements the composite training loss from:
  Abubakkar et al., "Interpretable Physics-Informed Load Forecasting for
  U.S. Grid Resilience", IEEE, 2026.

Loss formulation (Eq. 5):
    L_b = L_MSE + λ₁ · L_parabolic + λ₂ · L_ramp

Components
----------
L_MSE        : Standard mean-squared error (supervised term).
L_parabolic  : Penalises predictions outside the piecewise parabolic
               ERCOT temperature–demand envelope ± 2σ(T) tolerance band.
L_ramp       : Penalises hour-over-hour demand changes exceeding the
               99.5th-percentile ramp observed in training data (~4,800 MW/h).

ERCOT Parabolic Envelope Coefficients (Eq. 8, calibrated in [7]):
    T₀ = 18.5°C  (heating/cooling inflection point)
    Heating  regime (T < T₀): D(T) = 47.2·T² − 1560.6·T + 51230
    Cooling  regime (T ≥ T₀): D(T) = 52.4·T² − 864.5·T  + 35523.9
"""

import tensorflow as tf

# ── ERCOT piecewise parabolic envelope coefficients ──────────────────────────
# Calibrated on ERCOT 2018–2022 training data (Table I, [7])

T0 = 18.5  # °C  inflection point

# Heating regime (T < T0)
A1 = 47.2
B1 = -1560.6
C1 = 51_230.0

# Cooling regime (T >= T0)
A2 = 52.4
B2 = -864.5
C2 = 35_523.9

# Default physics-loss hyperparameters (tuned on validation set)
LAMBDA1_DEFAULT = 0.1   # parabolic constraint weight
LAMBDA2_DEFAULT = 0.05  # ramp constraint weight

# 99.5th-percentile hour-over-hour ramp in ERCOT 2018–2022 training set
DELTA_MAX_DEFAULT = 4_800.0  # MW/h


# ── Utility: piecewise parabolic envelope ────────────────────────────────────

def ercot_parabolic_envelope(temperature: tf.Tensor) -> tf.Tensor:
    """
    Compute D(T) — the ERCOT piecewise parabolic temperature–demand curve.

    D(T) = a₁T² + b₁T + c₁  if T < T₀   (heating regime)
           a₂T² + b₂T + c₂  if T ≥ T₀   (cooling regime)

    Parameters
    ----------
    temperature : tf.Tensor
        Air temperature values in °C, shape (N,).

    Returns
    -------
    tf.Tensor
        Fitted demand values D(T) in MW, shape (N,).
    """
    temperature = tf.cast(temperature, tf.float32)

    heating = A1 * temperature**2 + B1 * temperature + C1
    cooling = A2 * temperature**2 + B2 * temperature + C2

    return tf.where(temperature < T0, heating, cooling)


# ── Individual loss terms ────────────────────────────────────────────────────

def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Standard mean-squared error.

    L_MSE = (1/N) Σ (y_i − ŷ_i)²

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth demand values, shape (N,).
    y_pred : tf.Tensor
        Model predictions, shape (N,).

    Returns
    -------
    tf.Tensor
        Scalar MSE loss.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def parabolic_constraint_loss(
    y_pred: tf.Tensor,
    temperature: tf.Tensor,
    sigma_scale: float = 2.0,
    sigma_mw: float = 3_000.0,
) -> tf.Tensor:
    """
    Physics constraint penalising predictions outside the parabolic envelope.

    L_parabolic = (1/N) Σ max(0, |ŷ_i − D(T_i)| − ε(T_i))²

    where ε(T) = sigma_scale × σ(T) is the tolerance band derived from the
    two-sided prediction interval of the parabolic fit.

    Parameters
    ----------
    y_pred : tf.Tensor
        Predicted demand values, shape (N,).
    temperature : tf.Tensor
        Corresponding air temperature values in °C, shape (N,).
    sigma_scale : float
        Number of standard deviations defining the tolerance band (default: 2).
    sigma_mw : float
        Residual standard deviation of the parabolic fit in MW (default: 3000).
        Set to 2σ of training residuals; adjust from your calibration data.

    Returns
    -------
    tf.Tensor
        Scalar parabolic constraint loss.
    """
    D_T = ercot_parabolic_envelope(temperature)
    epsilon = sigma_scale * sigma_mw  # scalar tolerance in MW

    deviation = tf.abs(y_pred - D_T) - epsilon
    violation = tf.maximum(0.0, deviation)
    return tf.reduce_mean(tf.square(violation))


def ramp_constraint_loss(
    y_pred: tf.Tensor,
    delta_max: float = DELTA_MAX_DEFAULT,
) -> tf.Tensor:
    """
    Physics constraint penalising implausible hour-over-hour ramps.

    L_ramp = (1/N) Σ max(0, |ŷ_i − ŷ_{i−1}| − Δ_max)²

    Parameters
    ----------
    y_pred : tf.Tensor
        Predicted demand sequence, shape (N,).  Must be at least length 2.
    delta_max : float
        Maximum allowable hour-over-hour ramp in MW/h (default: 4800).
        Corresponds to the 99.5th percentile of training first differences.

    Returns
    -------
    tf.Tensor
        Scalar ramp constraint loss.
    """
    y_curr = y_pred[1:]   # ŷ_i
    y_prev = y_pred[:-1]  # ŷ_{i-1}

    ramp = tf.abs(y_curr - y_prev) - delta_max
    violation = tf.maximum(0.0, ramp)
    return tf.reduce_mean(tf.square(violation))


# ── Composite physics-informed loss ─────────────────────────────────────────

def physics_informed_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    temperature: tf.Tensor,
    lambda1: float = LAMBDA1_DEFAULT,
    lambda2: float = LAMBDA2_DEFAULT,
    delta_max: float = DELTA_MAX_DEFAULT,
    sigma_mw: float = 3_000.0,
) -> tf.Tensor:
    """
    Composite physics-informed training loss (Eq. 5).

    L_b = L_MSE + λ₁ · L_parabolic + λ₂ · L_ramp

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth demand, shape (N,).
    y_pred : tf.Tensor
        Model predictions, shape (N,).
    temperature : tf.Tensor
        Air temperature in °C, shape (N,).
    lambda1 : float
        Weight for parabolic constraint (default: 0.1).
    lambda2 : float
        Weight for ramp constraint (default: 0.05).
    delta_max : float
        Maximum allowable ramp in MW/h (default: 4800).
    sigma_mw : float
        Parabolic-fit residual standard deviation in MW (default: 3000).

    Returns
    -------
    tf.Tensor
        Scalar composite loss value.
    """
    l_mse = mse_loss(y_true, y_pred)
    l_para = parabolic_constraint_loss(y_pred, temperature, sigma_mw=sigma_mw)
    l_ramp = ramp_constraint_loss(y_pred, delta_max=delta_max)

    return l_mse + lambda1 * l_para + lambda2 * l_ramp


# ── Keras-compatible loss wrapper ────────────────────────────────────────────

class PhysicsInformedLoss(tf.keras.losses.Loss):
    """
    Keras Loss subclass wrapping the composite physics-informed loss.

    Usage
    -----
    loss_fn = PhysicsInformedLoss(lambda1=0.1, lambda2=0.05)
    model.compile(optimizer="adam", loss=loss_fn)

    Note: temperature must be passed as the last column of y_true when using
    this wrapper inside model.compile(), or use the functional API directly.
    """

    def __init__(
        self,
        lambda1: float = LAMBDA1_DEFAULT,
        lambda2: float = LAMBDA2_DEFAULT,
        delta_max: float = DELTA_MAX_DEFAULT,
        sigma_mw: float = 3_000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta_max = delta_max
        self.sigma_mw = sigma_mw

    def call(self, y_true, y_pred):
        """
        Expects y_true[:, 0] = demand, y_true[:, 1] = temperature (°C).
        """
        demand = tf.cast(y_true[:, 0], tf.float32)
        temperature = tf.cast(y_true[:, 1], tf.float32)
        y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)

        return physics_informed_loss(
            y_true=demand,
            y_pred=y_pred,
            temperature=temperature,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            delta_max=self.delta_max,
            sigma_mw=self.sigma_mw,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "delta_max": self.delta_max,
                "sigma_mw": self.sigma_mw,
            }
        )
        return config
