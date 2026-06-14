"""
feature_engineering.py
======================
Constructs the 27-feature multimodal feature matrix described in
Choudhary & Kulkarni (2026), Climatic Change §3.2.

Feature groups
--------------
1. Temporal encodings       : month_sin, month_cos, year_norm
2. CO₂ radiative forcing    : co2_ppm, co2_log, co2_yoy, co2_ma12/60,
                              radiative_forcing (RF = 5.35 × ln(C/C₀))
3. Solar cycle features     : sunspot_number, ssn_smooth11, ssn_zscore
4. Temperature lag features : temp_lag{1,2,3,6,12,24,36,60}
5. Rolling statistics       : temp_ma{6,12,24,36,60,120}, temp_std{…}
6. Trend & acceleration     : temp_trend, temp_accel, temp_ewm12/36
7. Interaction terms        : co2_x_ssn, forcing_trend

All features are standardised with RobustScaler fitted exclusively on the
training partition (1880–2017) to prevent data leakage.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

from utils import get_logger, project_root, ensure_dirs

warnings.filterwarnings("ignore")
logger = get_logger("feature_engineering")

# Pre-industrial CO₂ baseline (Myhre et al. 1998)
CO2_PREINDUSTRIAL = 280.0  # ppm

# Target column
TARGET = "temp_anomaly"

# Default train / val / test split years (paper §5.2)
TRAIN_END_YEAR = 2017
VAL_END_YEAR   = 2018   # val = 2017–2018; test = 2018–2025


# ─────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 27 physics-motivated predictive features to the merged dataset.

    Parameters
    ----------
    df : merged DataFrame with columns [date, temp_anomaly, co2_ppm, sunspot_number]

    Returns
    -------
    DataFrame with original columns + feature columns; NaN rows dropped.
    """
    df = df.copy()
    date = df["date"]

    # ── Group 1: Temporal encodings ──────────────────────────
    df["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    df["year_norm"] = (
        (date.dt.year - date.dt.year.min()) /
        (date.dt.year.max() - date.dt.year.min())
    )

    # ── Group 2: CO₂ radiative forcing ───────────────────────
    co2 = df["co2_ppm"].fillna(method="ffill")
    df["co2_ppm"]          = co2
    df["co2_log"]          = np.log(co2 / CO2_PREINDUSTRIAL)
    df["co2_yoy"]          = co2.pct_change(12) * 100
    df["co2_ma12"]         = co2.rolling(12).mean()
    df["co2_ma60"]         = co2.rolling(60).mean()
    # IPCC radiative forcing: Myhre et al. 1998 (used in paper)
    df["radiative_forcing"] = 5.35 * np.log(co2 / CO2_PREINDUSTRIAL)

    # ── Group 3: Solar cycle features ────────────────────────
    ssn = df["sunspot_number"].fillna(0)
    df["sunspot_number"] = ssn
    df["ssn_smooth11"]   = ssn.rolling(11 * 12, min_periods=1).mean()
    # Z-score relative to the rolling 11-year (Schwabe cycle) mean & std
    roll_mean = ssn.rolling(132).mean()
    roll_std  = ssn.rolling(132).std() + 1e-6
    df["ssn_zscore"]     = (ssn - roll_mean) / roll_std

    # ── Group 4: Temperature lag features ────────────────────
    temp = df[TARGET]
    for lag in [1, 2, 3, 6, 12, 24, 36, 60]:
        df[f"temp_lag{lag}"] = temp.shift(lag)

    # ── Group 5: Rolling statistics ──────────────────────────
    for w in [6, 12, 24, 36, 60, 120]:
        df[f"temp_ma{w}"]  = temp.rolling(w).mean()
        df[f"temp_std{w}"] = temp.rolling(w).std()

    # ── Group 6: Trend & acceleration ────────────────────────
    df["temp_trend"] = temp.diff(12)        # year-over-year change
    df["temp_accel"] = df["temp_trend"].diff(12)  # second derivative
    df["temp_ewm12"] = temp.ewm(span=12).mean()
    df["temp_ewm36"] = temp.ewm(span=36).mean()

    # ── Group 7: Interaction features ────────────────────────
    df["co2_x_ssn"]     = df["co2_ppm"] * df["ssn_zscore"]
    df["forcing_trend"] = df["radiative_forcing"] * df["year_norm"]

    # Drop rows with NaN (from lag creation at start of series)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        f"Feature matrix built: {df.shape[0]:,} rows × {df.shape[1]} cols"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (all except date, year, month, target)."""
    exclude = {"date", "year", "month", TARGET}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────
# Train / val / test split
# ─────────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_YEAR,
    val_end:   int = VAL_END_YEAR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split by year boundary (no shuffling).

    Paper §5.2:
        Train  : 1880–2017
        Val    : 2017–2018  (12 months)
        Test   : 2018–2025  (84 months)
    """
    year = df["date"].dt.year
    train = df[year <= train_end].copy()
    val   = df[(year > train_end) & (year <= val_end)].copy()
    test  = df[year > val_end].copy()

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        logger.info(
            f"{name:5s}: {len(part):4d} rows  "
            f"({part.date.dt.year.min()}–{part.date.dt.year.max()})"
        )
    return train, val, test


# ─────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────

class ClimateScaler:
    """
    Wraps RobustScaler (features) + MinMaxScaler (target) fitted only
    on the training split to prevent data leakage.
    """

    def __init__(self) -> None:
        self.feat_scaler   = RobustScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feat_cols: List[str] = []

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        feat_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on training data and return scaled (X_train, y_train)."""
        self.feat_cols = feat_cols or get_feature_columns(train_df)
        X_train = self.feat_scaler.fit_transform(train_df[self.feat_cols].values)
        y_train = self.target_scaler.fit_transform(
            train_df[[TARGET]].values
        ).ravel()
        logger.info(
            f"Scaler fitted on {len(train_df):,} training rows.  "
            f"Features: {len(self.feat_cols)}"
        )
        return X_train, y_train

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a val/test split using previously fitted scalers."""
        X = self.feat_scaler.transform(df[self.feat_cols].values)
        y = self.target_scaler.transform(df[[TARGET]].values).ravel()
        return X, y

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled target predictions back to °C."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()


# ─────────────────────────────────────────────────────────────
# Sequence builder for RNN/Transformer models
# ─────────────────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequence builder.

    Each sample is a (seq_len × n_features) tensor.
    The target y[i] corresponds to the step immediately
    following the window X[i-seq_len : i].

    Parameters
    ----------
    X       : scaled feature matrix  (n_samples × n_features)
    y       : scaled target vector   (n_samples,)
    seq_len : lookback window in months (paper: 36)

    Returns
    -------
    X_seq : (n_sequences, seq_len, n_features)
    y_seq : (n_sequences,)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Convenience: full pipeline
# ─────────────────────────────────────────────────────────────

def prepare_all(
    df: pd.DataFrame,
    seq_len: int = 36,
    train_end: int = TRAIN_END_YEAR,
    val_end:   int = VAL_END_YEAR,
):
    """
    Run the full feature-engineering → split → scale → sequence pipeline.

    Returns
    -------
    dict with keys:
        seq_train, seq_val, seq_test :  (X, y) tuples of sequences
        scaler                        :  fitted ClimateScaler
        train_df, val_df, test_df     :  raw DataFrames
        feat_cols                     :  feature column names
    """
    df_feat = build_features(df)
    feat_cols = get_feature_columns(df_feat)

    train_df, val_df, test_df = time_split(df_feat, train_end, val_end)

    scaler = ClimateScaler()
    X_tr, y_tr = scaler.fit_transform(train_df, feat_cols)
    X_va, y_va = scaler.transform(val_df)
    X_te, y_te = scaler.transform(test_df)

    seq_train = make_sequences(X_tr, y_tr, seq_len)
    seq_val   = make_sequences(X_va, y_va, seq_len)
    seq_test  = make_sequences(X_te, y_te, seq_len)

    logger.info(
        f"Sequence shapes — "
        f"Train: {seq_train[0].shape}  "
        f"Val: {seq_val[0].shape}  "
        f"Test: {seq_test[0].shape}"
    )

    return {
        "seq_train": seq_train,
        "seq_val"  : seq_val,
        "seq_test" : seq_test,
        "scaler"   : scaler,
        "train_df" : train_df,
        "val_df"   : val_df,
        "test_df"  : test_df,
        "feat_cols": feat_cols,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing import load_all

    raw = load_all(use_cache=True)
    result = prepare_all(raw)
    df_feat = build_features(raw)

    out = project_root() / "data" / "processed"
    ensure_dirs(str(out))
    df_feat.to_csv(out / "climate_merged_features.csv", index=False)

    # Save split metadata
    for split, name in [(result["train_df"], "train"),
                        (result["val_df"], "val"),
                        (result["test_df"], "test")]:
        split.to_csv(out / f"split_{name}.csv", index=False)

    print(f"\nFeature matrix saved → {out / 'climate_merged_features.csv'}")
    print(f"Feature columns ({len(result['feat_cols'])}):\n  "
          + "\n  ".join(result["feat_cols"]))
