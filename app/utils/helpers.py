"""
app/utils/helpers.py
─────────────────────
Shared preprocessing helpers used by prediction routes.

Key responsibilities
────────────────────
1. Feature engineering  — mirrors the notebook's Section 3 exactly so that
   raw telemetry → feature vector is reproducible at inference time.
2. Sequence building    — converts a flat feature matrix into (1, seq_len, n_feat)
   arrays expected by LSTM / GAN.
3. Scaling              — applies the pre-fitted StandardScaler from training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from app.config import settings


# ══════════════════════════════════════════════════════════════
# Feature Engineering  (mirrors notebook Section 3)
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as the training notebook (Section 3).

    Parameters
    ----------
    df : DataFrame with at minimum the raw metric columns produced by the pivot:
         voltageA, voltageB, voltageC,
         current1A, current1B, current1C,
         activePower1A, activePower1B, activePower1C,
         frequency,
         bucket_5m  (datetime),
         site_id    (int)

    Returns
    -------
    df : DataFrame enriched with all engineered features.
         Rows with NaN (from rolling / lag operations) are NOT dropped here —
         the caller decides how to handle them.
    """
    HORIZON = settings.sequence_len  # rolling window matches training

    # ── 3.1 Physical aggregates ───────────────────────────────
    df["voltage_mean"] = df[["voltageA", "voltageB", "voltageC"]].mean(axis=1)
    df["current_mean"] = df[["current1A", "current1B", "current1C"]].mean(axis=1)
    df["power_mean"]   = df[["activePower1A", "activePower1B", "activePower1C"]].mean(axis=1)

    df["voltage_std"]  = df[["voltageA", "voltageB", "voltageC"]].std(axis=1)
    df["current_std"]  = df[["current1A", "current1B", "current1C"]].std(axis=1)
    df["power_std"]    = df[["activePower1A", "activePower1B", "activePower1C"]].std(axis=1)

    # ── 3.2 Phase imbalance ───────────────────────────────────
    df["voltage_imbalance"] = df["voltage_std"] / (df["voltage_mean"] + 1e-9)
    df["current_imbalance"] = df["current_std"] / (df["current_mean"] + 1e-9)

    df["voltage_AB"] = df["voltageA"] - df["voltageB"]
    df["voltage_BC"] = df["voltageB"] - df["voltageC"]
    df["voltage_CA"] = df["voltageC"] - df["voltageA"]

    # ── 3.3 Apparent power & power factor ─────────────────────
    df["apparent_power_A"]     = df["voltageA"] * df["current1A"]
    df["apparent_power_B"]     = df["voltageB"] * df["current1B"]
    df["apparent_power_C"]     = df["voltageC"] * df["current1C"]
    df["apparent_power_total"] = df[["apparent_power_A",
                                     "apparent_power_B",
                                     "apparent_power_C"]].sum(axis=1)
    df["power_factor"] = (
        df["power_mean"] / (df["apparent_power_total"] / 3 + 1e-9)
    ).clip(0, 1.2)

    # ── 3.4 Frequency deviation ───────────────────────────────
    df["freq_deviation"] = (df["frequency"] - 50).abs()
    df["freq_deviation_roll"] = (
        df.groupby("site_id")["freq_deviation"]
        .rolling(HORIZON).mean()
        .reset_index(level=0, drop=True)
    )

    # ── 3.5 Rolling statistics ────────────────────────────────
    for col in ["voltage_mean", "power_mean", "current_mean"]:
        df[f"{col}_roll_mean"] = (
            df.groupby("site_id")[col]
            .rolling(HORIZON).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_roll_std"] = (
            df.groupby("site_id")[col]
            .rolling(HORIZON).std()
            .reset_index(level=0, drop=True)
        )

    # ── 3.6 EMA + deviation ───────────────────────────────────
    for col in ["voltage_mean", "power_mean"]:
        df[f"{col}_ema"] = (
            df.groupby("site_id")[col]
            .transform(lambda x: x.ewm(span=HORIZON, adjust=False).mean())
        )
        df[f"{col}_ema_diff"] = df[col] - df[f"{col}_ema"]

    # ── 3.7 Rate of change ────────────────────────────────────
    for col in ["voltage_mean", "power_mean", "frequency"]:
        df[f"{col}_roc"]  = df.groupby("site_id")[col].diff(1)
        df[f"{col}_roc2"] = df.groupby("site_id")[col].diff(2)

    # ── 3.8 Lag features ──────────────────────────────────────
    for lag in [1, 2, 3]:
        for col in ["voltage_mean", "power_mean"]:
            df[f"{col}_lag{lag}"] = df.groupby("site_id")[col].shift(lag)

    # ── 3.9 Consecutive low-voltage streak ───────────────────
    low_v = (df["voltage_mean"] < 200).astype(int)
    df["low_voltage_streak"] = (
        low_v.groupby(df["site_id"])
        .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    )

    # ── 3.10 Calendar / cyclical features ────────────────────
    df["hour"]        = df["bucket_5m"].dt.hour
    df["day_of_week"] = df["bucket_5m"].dt.dayofweek
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)

    return df


# ══════════════════════════════════════════════════════════════
# Preprocessing for tabular models (XGBoost / LightGBM)
# ══════════════════════════════════════════════════════════════

def preprocess_tabular(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler,
) -> np.ndarray:
    """
    Apply feature engineering + scaling and return a 2-D array
    ready for sklearn-compatible predict_proba().

    Parameters
    ----------
    df            : raw telemetry DataFrame (already pivoted, one row per timestep)
    feature_names : ordered list of top-N feature names from training
    scaler        : fitted StandardScaler

    Returns
    -------
    X : np.ndarray of shape (n_rows, n_features)
    """
    df = engineer_features(df.copy())
    df.dropna(inplace=True)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")

    X = scaler.transform(df[feature_names].values.astype(np.float32))
    return X


# ══════════════════════════════════════════════════════════════
# Sequence builder for LSTM / GAN
# ══════════════════════════════════════════════════════════════

def build_single_sequence(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler,
    seq_len: int,
) -> Tuple[np.ndarray, bool]:
    """
    Build a single (1, seq_len, n_feat) array from the last `seq_len` rows
    of `df`.  Returns (array, ok) where ok=False when there are not enough rows.

    Parameters
    ----------
    df            : raw telemetry (≥ seq_len + rolling_window rows recommended)
    feature_names : ordered top-N feature list from training
    scaler        : fitted StandardScaler
    seq_len       : number of timesteps per sequence (e.g. 12)
    """
    df = engineer_features(df.copy())
    df.dropna(inplace=True)

    if len(df) < seq_len:
        return np.array([]), False

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")

    window = df[feature_names].values[-seq_len:].astype(np.float32)
    window_scaled = scaler.transform(window)              # (seq_len, n_feat)
    return window_scaled[np.newaxis, ...], True           # (1, seq_len, n_feat)


# ══════════════════════════════════════════════════════════════
# Probability → prediction
# ══════════════════════════════════════════════════════════════

def apply_threshold(probability: float, threshold: float) -> int:
    return int(probability >= threshold)
