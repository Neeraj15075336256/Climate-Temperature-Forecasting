"""
utils.py
========
Shared utility functions: random seeding, file I/O, plotting helpers,
and logging for the Climate-Temperature-Forecasting pipeline.

Reference: Choudhary & Kulkarni (2026), Climatic Change.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def get_logger(name: str = "climate", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with timestamp prefix."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s | %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds for full reproducibility (Python, NumPy, TF)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info(f"Global seed set to {seed} (Python / NumPy / TensorFlow).")
    except ImportError:
        logger.warning("TensorFlow not available; seeded Python and NumPy only.")


# ─────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the repository root (parent of /src)."""
    return Path(__file__).resolve().parent.parent


def output_path(subfolder: str = "results/figures") -> Path:
    root = project_root()
    p = root / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────────────────────

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV with sensible defaults and error reporting."""
    try:
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise


def summarise_df(df: pd.DataFrame, label: str = "") -> None:
    """Print a concise summary of a DataFrame."""
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Shape    : {df.shape}")
    print(f"{tag}Columns  : {list(df.columns)}")
    print(f"{tag}Dtypes   :\n{df.dtypes.to_string()}")
    print(f"{tag}Nulls    :\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"{tag}Date range: {df['date'].min()} → {df['date'].max()}"
          if 'date' in df.columns else "")


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

# Publication-quality style
STYLE_PARAMS = {
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
}

PALETTE = {
    "cnn_lstm"    : "#1f77b4",
    "lstm"        : "#ff7f0e",
    "bilstm"      : "#2ca02c",
    "gru"         : "#d62728",
    "transformer" : "#9467bd",
    "lightgbm"    : "#8c564b",
    "xgboost"     : "#e377c2",
    "rf"          : "#7f7f7f",
    "ridge"       : "#bcbd22",
    "ensemble"    : "#000000",
    "paris"       : "#d62728",
    "warm"        : "#d62728",
    "cool"        : "#1f77b4",
}


def apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_PARAMS)


def save_figure(fig: plt.Figure, filename: str,
                subfolder: str = "results/figures") -> str:
    """Save figure to the results/figures directory."""
    out = output_path(subfolder) / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved → {out}")
    return str(out)


def plot_time_series(
    dates: pd.Series,
    values: pd.Series,
    title: str = "",
    ylabel: str = "Temperature Anomaly (°C)",
    ma_window: int = 60,
    figsize: Tuple[int, int] = (14, 5),
    save_as: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a climate time series with warm/cool shading and moving average.

    Parameters
    ----------
    dates     : DatetimeSeries index
    values    : anomaly values
    title     : plot title
    ylabel    : y-axis label
    ma_window : rolling mean window (months)
    figsize   : figure size tuple
    save_as   : filename to save (None = don't save)
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    ma = values.rolling(ma_window, center=True).mean()
    ax.fill_between(dates, values, where=values >= 0,
                    color=PALETTE["warm"], alpha=0.6, label="Warm anomaly")
    ax.fill_between(dates, values, where=values < 0,
                    color=PALETTE["cool"], alpha=0.6, label="Cool anomaly")
    ax.plot(dates, ma, "k-", lw=1.8, label=f"{ma_window//12}-yr moving avg")
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1.5, color=PALETTE["paris"], lw=1.2, ls=":",
               alpha=0.8, label="Paris 1.5°C")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


def plot_forecast_comparison(
    future_dates: pd.DatetimeIndex,
    forecasts: Dict[str, np.ndarray],
    ensemble: np.ndarray,
    hist_dates: pd.Series,
    hist_values: pd.Series,
    save_as: Optional[str] = "fig10_decadal_forecast.png",
) -> plt.Figure:
    """
    Plot 10-year decadal temperature forecast for all models + ensemble.
    Reproduces Figure 10 of the manuscript.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical context (last 20 years)
    mask = hist_dates >= (hist_dates.max() - pd.DateOffset(years=20))
    ax.plot(hist_dates[mask], hist_values[mask],
            color="grey", lw=1.2, alpha=0.85, label="Historical (Berkeley Earth)")

    colours = [PALETTE["cnn_lstm"], PALETTE["lstm"],
               PALETTE["bilstm"], PALETTE["gru"], PALETTE["transformer"]]

    for (name, fc), col in zip(forecasts.items(), colours):
        ax.plot(future_dates, fc, color=col, lw=1.4, ls="--",
                alpha=0.85, label=name)

    ax.plot(future_dates, ensemble, color=PALETTE["ensemble"],
            lw=2.5, label="Ensemble Mean")
    ax.axhline(1.5, color=PALETTE["paris"], ls=":", lw=1.5, alpha=0.9,
               label="Paris 1.5°C threshold")

    # Shade between ±1σ of ensemble
    fc_stack = np.stack(list(forecasts.values()), axis=0)
    mu  = fc_stack.mean(axis=0)
    std = fc_stack.std(axis=0)
    ax.fill_between(future_dates, mu - std, mu + std,
                    color="grey", alpha=0.15, label="Ensemble ±1σ")

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title("10-Year Global Temperature Anomaly Forecast (2025–2035)\n"
                 "Multimodal Machine Learning Ensemble", fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ─────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────

def save_json(obj: dict, path: str) -> None:
    """Serialise a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info(f"JSON saved → {path}")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Murphy Skill Score
# ─────────────────────────────────────────────────────────────

def murphy_skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                       y_clim: Optional[np.ndarray] = None) -> float:
    """
    Murphy (1987) Skill Score relative to climatological baseline.

    MSS = 1 - RMSE(model) / RMSE(climatology)
    MSS > 0  → model beats persistence climatology
    MSS > 0.5 → meaningful forecast skill

    Parameters
    ----------
    y_true  : observed values
    y_pred  : model predictions
    y_clim  : climatological predictions (default: training mean)
    """
    from sklearn.metrics import mean_squared_error
    if y_clim is None:
        y_clim = np.full_like(y_true, y_true.mean())
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_clim  = np.sqrt(mean_squared_error(y_true, y_clim))
    if rmse_clim == 0:
        return 0.0
    return float(1.0 - rmse_model / rmse_clim)
