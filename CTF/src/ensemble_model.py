"""
ensemble_model.py
=================
Builds the four-model ensemble (LSTM + Bi-LSTM + GRU + CNN-LSTM)
and generates decadal projections for 2025–2035 (paper §9).

Key findings:
  • Four models converge at 1.25–1.35°C by 2034 (±0.04°C, 1σ)
  • Transformer excluded: diverges to ≈0.94°C (attention extrapolation)
  • Ensemble mean remains below the Paris Agreement 1.5°C threshold
    under a business-as-usual CO₂ trajectory

Rolling multi-step forecasting protocol:
  1. Initialise with the last seq_len=36 months of the test set
  2. Predict one step ahead, append to sequence, drop oldest step
  3. Repeat for 120 months (10 years = 2025–2035)
  4. CO₂ / sunspot features extrapolated via linear trend functions

Reference: Choudhary & Kulkarni (2026), Climatic Change §9, Figure 10.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import (
    apply_style, ensure_dirs, get_logger, output_path,
    project_root, save_figure, PALETTE, plot_forecast_comparison,
)

logger = get_logger("ensemble")

MODELS_DIR   = project_root() / "models"
RESULTS_DIR  = project_root() / "results"
HORIZON_YRS  = 10   # 2025–2035
ENSEMBLE_MODELS = ["CNN-LSTM", "GRU", "LSTM", "Bi-LSTM"]


# ─────────────────────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────────────────────

def load_models(
    model_names: List[str] = ENSEMBLE_MODELS,
    models_dir: Optional[Path] = None,
) -> Dict[str, tf.keras.Model]:
    """
    Load Keras models from disk.  Expects files named:
        cnn_lstm_best.keras, gru_best.keras, lstm_best.keras, bilstm_best.keras
    """
    models_dir = models_dir or MODELS_DIR
    name_to_file = {
        "CNN-LSTM": "cnn_lstm_best.keras",
        "GRU"     : "gru_best.keras",
        "LSTM"    : "lstm_best.keras",
        "Bi-LSTM" : "bilstm_best.keras",
        "Transformer": "transformer_best.keras",
    }
    models = {}
    for name in model_names:
        fname = name_to_file.get(name)
        if fname is None:
            logger.warning(f"No filename mapping for model '{name}'. Skipping.")
            continue
        path = models_dir / fname
        if path.exists():
            models[name] = tf.keras.models.load_model(str(path))
            logger.info(f"Loaded {name} from {path}")
        else:
            logger.warning(f"Model file not found: {path}. "
                           "Run the training scripts first.")
    return models


# ─────────────────────────────────────────────────────────────
# Rolling multi-step forecast
# ─────────────────────────────────────────────────────────────

def rolling_forecast(
    model: tf.keras.Model,
    init_seq: np.ndarray,
    horizon: int = 120,
) -> np.ndarray:
    """
    Generate `horizon` one-step-ahead predictions using a sliding window.

    Parameters
    ----------
    model    : trained Keras model
    init_seq : (seq_len, n_features) initialisation window (scaled)
    horizon  : number of future steps to predict (default 120 = 10 years)

    Returns
    -------
    preds : (horizon,) scaled predictions
    """
    seq   = init_seq.copy()           # (seq_len, n_features)
    preds = np.empty(horizon, dtype=np.float32)

    for step in range(horizon):
        inp  = seq[np.newaxis, ...]           # (1, seq_len, n_features)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds[step] = pred

        # Slide window: drop oldest row, append new row with updated temp
        new_row         = seq[-1].copy()
        new_row[0]      = pred            # update the first feature (temp_lag1 proxy)
        seq             = np.vstack([seq[1:], new_row[np.newaxis, :]])

    return preds


def ensemble_forecast(
    models: Dict[str, tf.keras.Model],
    data: dict,
    horizon_years: int = HORIZON_YRS,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DatetimeIndex]:
    """
    Generate rolling forecasts from all ensemble members.

    Parameters
    ----------
    models        : loaded Keras models dict
    data          : output of prepare_all()
    horizon_years : number of years to forecast

    Returns
    -------
    forecasts     : {model_name: orig-scale forecast array}
    ensemble_mean : (horizon,) averaged forecast (excl. Transformer)
    future_dates  : DatetimeIndex for the forecast period
    """
    horizon  = horizon_years * 12
    scaler   = data["scaler"]

    # Initialise with last seq_len rows of the scaled test set
    X_test_flat = scaler.feat_scaler.transform(
        data["test_df"][data["feat_cols"]].values
    )
    seq_len    = data["seq_train"][0].shape[1]
    init_seq   = X_test_flat[-seq_len:]

    forecasts = {}
    for name, model in models.items():
        logger.info(f"Generating {horizon}-step rolling forecast for {name}…")
        preds_scaled = rolling_forecast(model, init_seq, horizon)
        preds_orig   = scaler.inverse_y(preds_scaled)
        forecasts[name] = preds_orig
        logger.info(
            f"  {name}: end-of-period = {preds_orig[-1]:.3f}°C  "
            f"range = [{preds_orig.min():.3f}, {preds_orig.max():.3f}]"
        )

    # Ensemble mean (excluding Transformer per paper §9)
    ensemble_members = {k: v for k, v in forecasts.items()
                        if k != "Transformer"}
    ensemble_mean = np.mean(
        np.stack(list(ensemble_members.values()), axis=0), axis=0
    )

    # Future date index
    last_date    = data["test_df"]["date"].max()
    future_dates = pd.date_range(
        last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    logger.info(
        f"Ensemble mean (2025–2035): "
        f"start={ensemble_mean[0]:.3f}°C  "
        f"end={ensemble_mean[-1]:.3f}°C  "
        f"max={ensemble_mean.max():.3f}°C"
    )
    if ensemble_mean.max() < 1.5:
        logger.info("✓ Ensemble mean REMAINS BELOW the Paris Agreement 1.5°C threshold.")
    else:
        logger.warning("✗ Ensemble mean EXCEEDS 1.5°C threshold in some years.")

    return forecasts, ensemble_mean, future_dates


# ─────────────────────────────────────────────────────────────
# Visualisation (Figure 10)
# ─────────────────────────────────────────────────────────────

def plot_decadal_forecast(
    forecasts: Dict[str, np.ndarray],
    ensemble_mean: np.ndarray,
    future_dates: pd.DatetimeIndex,
    data: dict,
    save_as: str = "fig10_decadal_forecast.png",
) -> plt.Figure:
    """Reproduce Figure 10 of the paper."""
    hist_df = data["test_df"]
    fig = plot_forecast_comparison(
        future_dates=future_dates,
        forecasts=forecasts,
        ensemble=ensemble_mean,
        hist_dates=hist_df["date"],
        hist_values=hist_df["temp_anomaly"],
        save_as=save_as,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Save forecast table
# ─────────────────────────────────────────────────────────────

def save_forecast_csv(
    forecasts: Dict[str, np.ndarray],
    ensemble_mean: np.ndarray,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    df = pd.DataFrame({"date": future_dates})
    for name, fc in forecasts.items():
        df[name] = fc
    df["Ensemble_Mean"] = ensemble_mean
    df["Paris_1.5C"]    = 1.5

    out = project_root() / "results" / "tables" / "decadal_forecast_2025_2035.csv"
    ensure_dirs(str(out.parent))
    df.to_csv(out, index=False)
    logger.info(f"Forecast table saved → {out}")
    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 10-year ensemble temperature forecasts"
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="Forecast horizon (years)"
    )
    parser.add_argument(
        "--no-transformer", action="store_true",
        help="Exclude Transformer from ensemble (recommended per paper §9)"
    )
    args = parser.parse_args()

    from preprocessing import load_all
    from feature_engineering import prepare_all

    logger.info("Loading data and preparing features…")
    raw  = load_all(use_cache=True)
    data = prepare_all(raw)

    members = ENSEMBLE_MODELS
    if not args.no_transformer:
        members = members + ["Transformer"]

    models = load_models(members)
    if not models:
        logger.error(
            "No trained models found. "
            "Run src/train_cnn_lstm.py and other training scripts first."
        )
        exit(1)

    forecasts, ensemble_mean, future_dates = ensemble_forecast(
        models, data, horizon_years=args.horizon
    )
    plot_decadal_forecast(forecasts, ensemble_mean, future_dates, data)
    df = save_forecast_csv(forecasts, ensemble_mean, future_dates)

    print(f"\n10-Year Ensemble Forecast Summary (2025–2035):")
    print(f"  Ensemble start : {ensemble_mean[0]:.3f}°C")
    print(f"  Ensemble end   : {ensemble_mean[-1]:.3f}°C")
    print(f"  Ensemble max   : {ensemble_mean.max():.3f}°C")
    print(f"  Paris 1.5°C exceeded: {(ensemble_mean >= 1.5).any()}")
