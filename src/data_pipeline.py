"""
Data Preprocessing Pipeline for ERCOT + ASOS Dataset.

Handles ingestion, merging, feature engineering, normalisation, and
sliding-window sequence construction for the physics-informed ensemble.

Dataset description (Table I of the paper):
  - ERCOT hourly system load: 2018–2025 (70,128 observations)
  - ASOS meteorological data: 3 Texas stations (BKS, JDD, TME)
  - Missing weather: <0.6% — filled via linear interpolation
  - Temporal splits:
      Train : 2018–2022  (43,824 h)
      Val   : 2023       ( 8,760 h)
      Test  : 2024–2025  (17,544 h)

Input features (F = 12):
    [lagged_load, air_temp, feels_like, humidity, wind_speed,
     precip, w_data, hour, day_of_week, month, weekend_flag, holiday_flag]

Target: next-hour demand y_{t+1} (MW).
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# U.S. federal holidays (static list; extend as needed)
US_FEDERAL_HOLIDAYS = {
    # Format: (month, day) — fixed-date holidays only
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (11, 11), # Veterans Day
    (12, 25), # Christmas Day
}


# ── Utility functions ────────────────────────────────────────────────────────

def is_federal_holiday(dt: pd.Timestamp) -> int:
    """Return 1 if date is a U.S. federal holiday (simplified), else 0."""
    return int((dt.month, dt.day) in US_FEDERAL_HOLIDAYS)


def feels_like_temperature(
    temp_c: pd.Series,
    humidity: pd.Series,
    wind_speed_ms: pd.Series,
) -> pd.Series:
    """
    Compute apparent temperature using a combined heat-index / wind-chill
    approximation.

    - Heat index is applied when T ≥ 27°C and RH ≥ 40%.
    - Wind chill is applied when T ≤ 10°C and wind speed > 1.3 m/s.
    - Elsewhere, apparent temperature equals dry-bulb temperature.

    Parameters
    ----------
    temp_c : pd.Series   — air temperature in °C
    humidity : pd.Series — relative humidity in %
    wind_speed_ms : pd.Series — wind speed in m/s

    Returns
    -------
    pd.Series — apparent temperature in °C
    """
    T = temp_c
    RH = humidity
    V = wind_speed_ms * 3.6  # m/s → km/h for wind-chill formula

    # Heat index (°C) — simplified Rothfusz regression
    HI = (
        -8.784695
        + 1.61139411 * T
        + 2.338549 * RH
        - 0.14611605 * T * RH
        - 0.012308094 * T**2
        - 0.016424828 * RH**2
        + 0.002211732 * T**2 * RH
        + 0.00072546 * T * RH**2
        - 0.000003582 * T**2 * RH**2
    )

    # Wind chill (°C) — Environment Canada formula (T in °C, V in km/h)
    WC = (
        13.12
        + 0.6215 * T
        - 11.37 * V**0.16
        + 0.3965 * T * V**0.16
    )

    result = T.copy()
    heat_mask = (T >= 27) & (RH >= 40)
    chill_mask = (T <= 10) & (V > 4.8)  # 4.8 km/h ≈ 1.3 m/s
    result[heat_mask] = HI[heat_mask]
    result[chill_mask] = WC[chill_mask]

    return result


# ── ERCOT loader ─────────────────────────────────────────────────────────────

def load_ercot(path: str) -> pd.DataFrame:
    """
    Load and standardise ERCOT hourly load CSV.

    Expected columns (flexible): timestamp column + a demand column
    (MW).  The function attempts to auto-detect the timestamp and demand
    columns by keyword matching.

    Parameters
    ----------
    path : str
        Path to the ERCOT CSV file.

    Returns
    -------
    pd.DataFrame
        Indexed by hourly UTC timestamp with column 'demand_mw'.
    """
    df = pd.read_csv(path)

    # ── Auto-detect timestamp column ─────────────────────────────────────────
    ts_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ("hour", "date", "time", "interval", "dst")
    )]
    if not ts_candidates:
        raise ValueError(
            f"Could not find a timestamp column in {path}. "
            "Expected a column containing 'hour', 'date', 'time', or 'interval'."
        )
    ts_col = ts_candidates[0]

    # ── Auto-detect demand column ─────────────────────────────────────────────
    demand_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ("total", "load", "demand", "mw", "ercot")
    ) and c != ts_col]
    if not demand_candidates:
        raise ValueError(
            f"Could not find a demand column in {path}. "
            "Expected a column containing 'total', 'load', 'demand', or 'mw'."
        )
    demand_col = demand_candidates[0]

    df = df[[ts_col, demand_col]].copy()
    df.columns = ["timestamp", "demand_mw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    df = df.sort_index()
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

    return df


# ── ASOS loader ───────────────────────────────────────────────────────────────

def load_asos(path: str) -> pd.DataFrame:
    """
    Load and standardise a single ASOS station CSV (IEM format).

    Expected IEM columns include: 'valid', 'tmpf' (°F), 'relh', 'sknt'
    (knots), 'p01i' (precipitation inches), 'wxcodes'.

    Parameters
    ----------
    path : str
        Path to ASOS station CSV (downloaded from mesonet.agron.iastate.edu).

    Returns
    -------
    pd.DataFrame
        Hourly resampled weather DataFrame indexed by UTC timestamp.
    """
    df = pd.read_csv(path, low_memory=False)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    if "valid" in df.columns:
        df["timestamp"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    else:
        ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # ── Meteorological variables (convert to SI) ──────────────────────────────
    wx = pd.DataFrame(index=df.index)

    # Temperature: °F → °C
    if "tmpf" in df.columns:
        wx["air_temp"] = (pd.to_numeric(df["tmpf"], errors="coerce") - 32) * 5 / 9
    elif "tmpc" in df.columns:
        wx["air_temp"] = pd.to_numeric(df["tmpc"], errors="coerce")
    else:
        warnings.warn("No temperature column found in ASOS file.")
        wx["air_temp"] = np.nan

    # Relative humidity (%)
    wx["humidity"] = pd.to_numeric(df.get("relh", np.nan), errors="coerce")

    # Wind speed: knots → m/s (1 knot = 0.514444 m/s)
    if "sknt" in df.columns:
        wx["wind_speed"] = pd.to_numeric(df["sknt"], errors="coerce") * 0.514444
    else:
        wx["wind_speed"] = np.nan

    # Precipitation: inches → mm (1 inch = 25.4 mm)
    if "p01i" in df.columns:
        wx["precip"] = pd.to_numeric(df["p01i"], errors="coerce") * 25.4
    else:
        wx["precip"] = np.nan

    # Weather type code (encoded as integer categories)
    if "wxcodes" in df.columns:
        wx["w_data"] = pd.Categorical(df["wxcodes"].fillna("CLR")).codes.astype(float)
    else:
        wx["w_data"] = 0.0

    # Resample to hourly using mean (to combine sub-hourly observations)
    wx = wx.resample("1H").mean()

    return wx


# ── Multi-station ASOS merger ─────────────────────────────────────────────────

def load_and_merge_asos_stations(paths: list) -> pd.DataFrame:
    """
    Load multiple ASOS station files and average meteorological variables.

    Parameters
    ----------
    paths : list of str
        File paths for each ASOS station (BKS, JDD, TME).

    Returns
    -------
    pd.DataFrame
        Hourly mean meteorological variables across all stations.
    """
    dfs = [load_asos(p) for p in paths]
    combined = pd.concat(dfs).groupby(level=0).mean()
    combined = combined.sort_index()
    return combined


# ── Main feature engineering pipeline ────────────────────────────────────────

def build_features(
    ercot_df: pd.DataFrame,
    asos_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ERCOT and ASOS data and engineer all 12 model features.

    Parameters
    ----------
    ercot_df : pd.DataFrame
        ERCOT demand, indexed by hourly UTC timestamp, column 'demand_mw'.
    asos_df : pd.DataFrame
        ASOS meteorological data, indexed by hourly UTC timestamp.

    Returns
    -------
    pd.DataFrame
        Combined feature DataFrame with columns:
        [demand_mw, lagged_load, air_temp, feels_like, humidity,
         wind_speed, precip, w_data, hour, day_of_week, month,
         weekend_flag, holiday_flag]
    """
    df = ercot_df.join(asos_df, how="inner")

    # ── Fill missing weather (<0.6% of records) via linear interpolation ──────
    weather_cols = ["air_temp", "humidity", "wind_speed", "precip", "w_data"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time").fillna(method="bfill")

    # ── Feels-like temperature ────────────────────────────────────────────────
    df["feels_like"] = feels_like_temperature(
        df["air_temp"], df["humidity"], df["wind_speed"]
    )

    # ── Lag-1 hourly demand ───────────────────────────────────────────────────
    df["lagged_load"] = df["demand_mw"].shift(1)

    # ── Calendar features ─────────────────────────────────────────────────────
    local_time = df.index.tz_convert("US/Central")
    df["hour"] = local_time.hour
    df["day_of_week"] = local_time.dayofweek + 1  # 1 = Monday
    df["month"] = local_time.month
    df["weekend_flag"] = (local_time.dayofweek >= 5).astype(int)
    df["holiday_flag"] = [is_federal_holiday(t) for t in local_time]

    # Drop first row (NaN lagged_load)
    df = df.dropna(subset=["lagged_load", "air_temp"])

    return df


# ── Sliding-window sequence builder ──────────────────────────────────────────

FEATURE_COLS = [
    "lagged_load", "air_temp", "feels_like", "humidity",
    "wind_speed", "precip", "w_data",
    "hour", "day_of_week", "month", "weekend_flag", "holiday_flag",
]


def make_sequences(
    df: pd.DataFrame,
    seq_len: int = 24,
    target_col: str = "demand_mw",
    feature_cols: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for model input.

    Each sample X_i ∈ ℝ^{T×F} is a 24-hour window; y_i is the next-hour
    demand; T_i is the temperature at the forecast horizon (for the
    physics-informed loss parabolic term).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from build_features().
    seq_len : int
        Sliding window length in hours (default: 24).
    target_col : str
        Column name of the target variable (default: 'demand_mw').
    feature_cols : list of str, optional
        Columns to use as model inputs (default: FEATURE_COLS).

    Returns
    -------
    X : np.ndarray, shape (N, seq_len, F)
    y : np.ndarray, shape (N,)  — next-hour demand targets
    T_forecast : np.ndarray, shape (N,) — temperature at forecast horizon
    """
    feature_cols = feature_cols or FEATURE_COLS
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    temps = df["air_temp"].values.astype(np.float32)

    N = len(features) - seq_len
    X = np.stack([features[i: i + seq_len] for i in range(N)])
    y = targets[seq_len:]
    T_forecast = temps[seq_len:]

    return X, y, T_forecast


# ── Train / val / test splitter ───────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / validation / test partitions.

    Default split matches the paper:
      Train : 2018–2022
      Val   : 2023
      Test  : 2024–2025

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with a UTC DatetimeTzIndex.
    train_end : str
        Inclusive end date for training split.
    val_end : str
        Inclusive end date for validation split.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    train_df = df[df.index <= pd.Timestamp(train_end, tz="UTC")]
    val_df = df[
        (df.index > pd.Timestamp(train_end, tz="UTC"))
        & (df.index <= pd.Timestamp(val_end, tz="UTC"))
    ]
    test_df = df[df.index > pd.Timestamp(val_end, tz="UTC")]

    print(
        f"[Split] Train: {len(train_df):,} h | "
        f"Val: {len(val_df):,} h | "
        f"Test: {len(test_df):,} h"
    )
    return train_df, val_df, test_df


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on training sequences.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, T, F)

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fitted scaler (on reshaped N×F view of training data).
    """
    N, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply a fitted scaler to a 3-D input array.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, F)
    scaler : StandardScaler

    Returns
    -------
    np.ndarray, shape (N, T, F)
    """
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
