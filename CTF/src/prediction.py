"""
prediction.py
=============
Rolling multi-step prediction utilities for generating future
temperature anomaly forecasts beyond the test period.

Includes:
  - CO₂ trajectory extrapolation (linear + exponential options)
  - Sunspot cycle extrapolation (sinusoidal Schwabe-cycle model)
  - Rolling-window feature update for autoregressive forecasting
  - Uncertainty estimation via ensemble spread

Reference: Choudhary & Kulkarni (2026), Climatic Change §9.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from utils import get_logger

logger = get_logger("prediction")

# Pre-industrial CO₂ baseline for radiative forcing
CO2_PREINDUSTRIAL = 280.0   # ppm
SCHWABE_PERIOD    = 11.0    # years


# ─────────────────────────────────────────────────────────────
# CO₂ trajectory extrapolation
# ─────────────────────────────────────────────────────────────

def extrapolate_co2(
    co2_series: pd.Series,
    horizon: int = 120,
    method: str = "linear",
) -> np.ndarray:
    """
    Extrapolate monthly CO₂ concentration for `horizon` future months.

    Parameters
    ----------
    co2_series : historical monthly CO₂ (ppm), pd.Series
    horizon    : number of months to project
    method     : 'linear' (default) or 'exponential'

    Returns
    -------
    np.ndarray of shape (horizon,) with projected CO₂ values (ppm)

    Notes
    -----
    Linear extrapolation assumes a constant gradient equal to the
    mean annual increment over the last 10 years. The 2023 rate was
    ~2.4 ppm/yr (NOAA GML 2025), consistent with business-as-usual
    trajectories used in the paper (§9).
    """
    last_val = float(co2_series.iloc[-1])
    t        = np.arange(len(co2_series)).reshape(-1, 1)
    t_future = np.arange(len(co2_series), len(co2_series) + horizon)

    if method == "exponential":
        log_co2 = np.log(co2_series.values)
        reg = LinearRegression().fit(t, log_co2)
        co2_future = np.exp(reg.predict(t_future.reshape(-1, 1)))
    else:   # linear (default)
        # Use last 10 years for gradient estimation
        window = min(120, len(co2_series))
        recent = co2_series.iloc[-window:].values
        t_rec  = np.arange(window).reshape(-1, 1)
        reg    = LinearRegression().fit(t_rec, recent)
        slope  = reg.coef_[0]   # ppm per month
        co2_future = last_val + slope * np.arange(1, horizon + 1)

    # Add seasonal breathing cycle (Keeling Curve seasonal amplitude ≈3.5 ppm)
    t_month = np.arange(horizon)
    seasonal = 3.5 * np.sin(2 * np.pi * t_month / 12 + 4.2)
    co2_future += seasonal

    logger.info(
        f"CO₂ extrapolation ({method}): "
        f"{last_val:.1f} → {co2_future[-1]:.1f} ppm  "
        f"(+{co2_future[-1]-last_val:.1f} ppm over {horizon//12} yr)"
    )
    return co2_future.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Sunspot cycle extrapolation
# ─────────────────────────────────────────────────────────────

def extrapolate_sunspots(
    ssn_series: pd.Series,
    horizon: int = 120,
) -> np.ndarray:
    """
    Extrapolate monthly sunspot number using a sinusoidal Schwabe cycle model.

    The 11-year (~132-month) Schwabe cycle is fitted to the most recent
    two complete cycles (Cycles 24–25) and projected forward.

    Parameters
    ----------
    ssn_series : historical monthly sunspot numbers, pd.Series
    horizon    : number of months to project

    Returns
    -------
    np.ndarray of shape (horizon,) with projected sunspot values
    """
    # Use last 24 years (2 Schwabe cycles) for phase fitting
    window  = min(24 * 12, len(ssn_series))
    recent  = ssn_series.iloc[-window:].values
    period  = SCHWABE_PERIOD * 12   # months

    # Least-squares fit: A * |sin(π*t/T + φ)| + B
    t       = np.arange(window)
    amp     = (recent.max() - recent.min()) / 2
    baseline = recent.min()

    # Simple phase estimation via cross-correlation with ideal cycle
    ideal   = amp * np.abs(np.sin(np.pi * t / period))
    phase   = np.argmax(np.correlate(recent - baseline, ideal, mode="full")) - window + 1
    phase  %= period

    t_future = np.arange(window, window + horizon)
    ssn_future = (
        amp * np.abs(np.sin(np.pi * (t_future + phase) / period))
        + baseline
        + np.random.normal(0, 5, horizon)  # observational noise
    )
    ssn_future = np.clip(ssn_future, 0, 320).astype(np.float32)

    logger.info(
        f"Sunspot extrapolation: "
        f"amp≈{amp:.0f}  period≈{SCHWABE_PERIOD:.1f} yr  "
        f"mean={ssn_future.mean():.1f}"
    )
    return ssn_future


# ─────────────────────────────────────────────────────────────
# Feature-vector updater for rolling forecast
# ─────────────────────────────────────────────────────────────

class RollingPredictor:
    """
    Autoregressive rolling predictor for multi-step temperature forecasting.

    At each step:
    1. Predict the next temperature anomaly using the current window
    2. Update lag features using the new prediction
    3. Update CO₂ and sunspot features from the extrapolated trajectories
    4. Slide the window forward by one month

    Parameters
    ----------
    model        : trained Keras model
    init_window  : (seq_len, n_features) scaled feature matrix for initialisation
    co2_future   : (horizon,) extrapolated CO₂ values (original scale)
    ssn_future   : (horizon,) extrapolated sunspot values
    scaler       : ClimateScaler with inverse_y()
    feat_cols    : ordered list of feature column names
    """

    # Feature positions depend on the order in feature_engineering.py
    _CO2_FEATURES    = ["co2_ppm", "co2_log", "co2_ma12", "co2_ma60",
                         "radiative_forcing"]
    _SOLAR_FEATURES  = ["sunspot_number", "ssn_smooth11", "ssn_zscore"]
    _TEMP_LAG_PREFIX = "temp_lag"
    _TEMP_MA_PREFIX  = "temp_ma"

    def __init__(
        self,
        model: tf.keras.Model,
        init_window: np.ndarray,
        co2_future: np.ndarray,
        ssn_future: np.ndarray,
        scaler,
        feat_cols: List[str],
    ) -> None:
        self.model       = model
        self.window      = init_window.copy()   # (seq_len, n_features)
        self.co2_future  = co2_future
        self.ssn_future  = ssn_future
        self.scaler      = scaler
        self.feat_cols   = feat_cols
        self._col_idx    = {c: i for i, c in enumerate(feat_cols)}
        self._seq_len    = init_window.shape[0]
        self._history    = []    # unscaled predictions

    def _get_idx(self, name: str) -> Optional[int]:
        return self._col_idx.get(name)

    def predict_step(self, step: int) -> float:
        """Predict one step and advance the rolling window."""
        inp          = self.window[np.newaxis, ...]   # (1, seq_len, n_feat)
        pred_scaled  = float(self.model.predict(inp, verbose=0)[0, 0])
        pred_orig    = float(self.scaler.inverse_y(np.array([pred_scaled]))[0])
        self._history.append(pred_orig)

        # Build the new feature row by copying the last row and updating
        new_row = self.window[-1].copy()

        # Update CO₂ features (if available in feature set)
        if step < len(self.co2_future):
            co2   = float(self.co2_future[step])
            rf    = 5.35 * np.log(co2 / CO2_PREINDUSTRIAL)
            for fname, val in [
                ("co2_ppm", co2),
                ("co2_log", np.log(co2 / CO2_PREINDUSTRIAL)),
                ("radiative_forcing", rf),
            ]:
                idx = self._get_idx(fname)
                if idx is not None:
                    # Apply the same scaler transform as training
                    # (approximation: update raw position — precise scaling
                    #  would require re-fitting, which leaks information)
                    new_row[idx] = (co2 - 370) / 40   # rough standardisation

        # Update solar features
        if step < len(self.ssn_future):
            ssn = float(self.ssn_future[step])
            idx = self._get_idx("sunspot_number")
            if idx is not None:
                new_row[idx] = (ssn - 80) / 60   # rough standardisation
            idx11 = self._get_idx("ssn_smooth11")
            if idx11 is not None:
                new_row[idx11] = new_row[idx11] * 0.9 + (ssn - 80) / 60 * 0.1

        # Update the most recent temp_lag1 (position 0 approximation)
        lag1_idx = self._get_idx("temp_lag1")
        if lag1_idx is not None:
            new_row[lag1_idx] = pred_scaled

        # Slide window
        self.window = np.vstack([self.window[1:], new_row[np.newaxis, :]])
        return pred_orig

    def run(self, horizon: int) -> np.ndarray:
        """Run the rolling predictor for `horizon` steps."""
        for step in range(horizon):
            self.predict_step(step)
        return np.array(self._history, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Ensemble uncertainty
# ─────────────────────────────────────────────────────────────

def ensemble_uncertainty(
    forecasts: Dict[str, np.ndarray],
    exclude: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ensemble mean ± 1σ across member forecasts.

    Parameters
    ----------
    forecasts : {model_name: (horizon,) array}
    exclude   : model names to exclude (e.g. ['Transformer'])

    Returns
    -------
    mean, lower (mean-σ), upper (mean+σ)
    """
    exclude = exclude or []
    members = np.stack(
        [v for k, v in forecasts.items() if k not in exclude], axis=0
    )
    mu  = members.mean(axis=0)
    sig = members.std(axis=0)
    logger.info(
        f"Ensemble ({members.shape[0]} members): "
        f"final mean={mu[-1]:.3f}°C ± {sig[-1]:.3f}°C"
    )
    return mu, mu - sig, mu + sig


# ─────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with synthetic data
    logger.info("Running prediction module smoke test…")

    # Fake CO₂ series (recent upward trend)
    co2_hist = pd.Series(np.linspace(315, 422, 800))
    co2_proj = extrapolate_co2(co2_hist, horizon=120)
    logger.info(f"CO₂ projection range: {co2_proj.min():.1f}–{co2_proj.max():.1f} ppm")

    # Fake sunspot series
    t   = np.arange(800)
    ssn = 80 + 60 * np.abs(np.sin(np.pi * t / 132)) + np.random.exponential(8, 800)
    ssn_proj = extrapolate_sunspots(pd.Series(ssn), horizon=120)
    logger.info(f"SSN projection range: {ssn_proj.min():.1f}–{ssn_proj.max():.1f}")

    logger.info("Smoke test complete.")
