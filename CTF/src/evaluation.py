"""
evaluation.py
=============
Unified evaluation module for all nine models.

Metrics (paper §6.1, Table 4):
    MAE   — Mean Absolute Error         (°C)
    RMSE  — Root Mean Squared Error     (°C)
    MAPE  — Mean Absolute Percentage Error (%)
    R²    — Coefficient of determination
    MSS   — Murphy (1987) Skill Score   (>0.5 = meaningful forecast)

Residual diagnostics (paper §8):
    Q-Q plot, Residuals vs. Predicted, Residual histogram
    → verifies near-Gaussian error distribution for CNN-LSTM

Reference: Choudhary & Kulkarni (2026), Climatic Change §6 & §8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from utils import (
    apply_style, get_logger, murphy_skill_score, output_path, project_root,
    save_figure, ensure_dirs, PALETTE,
)

logger = get_logger("evaluation")


# ─────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_single_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    name: str = "Model",
    y_train_orig: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all five paper metrics for one trained model.

    Parameters
    ----------
    model        : trained Keras model (must have .predict)
    X_test       : (n_seq, seq_len, n_feat) test sequences (scaled)
    y_test       : (n_seq,) test targets (scaled)
    scaler       : ClimateScaler with .inverse_y()
    name         : model display name
    y_train_orig : original-scale training targets for climatology baseline

    Returns
    -------
    dict with MAE, RMSE, MAPE, R2, Skill, pred, true
    """
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred        = scaler.inverse_y(y_pred_scaled)
    y_true        = scaler.inverse_y(y_test)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = mean_absolute_percentage_error(y_true + 1e-6, y_pred + 1e-6) * 100
    r2    = r2_score(y_true, y_pred)

    if y_train_orig is not None:
        y_clim = np.full_like(y_true, y_train_orig.mean())
    else:
        y_clim = None
    skill = murphy_skill_score(y_true, y_pred, y_clim)

    logger.info(
        f"[{name}] MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"MAPE={mape:.2f}%  R²={r2:.4f}  Skill={skill:.4f}"
    )

    return {
        "Model": name,
        "MAE"  : mae,
        "RMSE" : rmse,
        "MAPE" : mape,
        "R2"   : r2,
        "Skill": skill,
        "pred" : y_pred,
        "true" : y_true,
    }


# ─────────────────────────────────────────────────────────────
# Multi-model benchmark table
# ─────────────────────────────────────────────────────────────

def benchmark_table(results: Dict[str, dict]) -> pd.DataFrame:
    """
    Build the nine-model benchmark table (Table 4 in the paper).

    Parameters
    ----------
    results : mapping of model_name → result dict from evaluate_single_model

    Returns
    -------
    Sorted DataFrame (ascending MAE), rounded to 4 d.p.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Model"     : name,
            "MAE (°C)"  : round(r["MAE"],   4),
            "RMSE (°C)" : round(r["RMSE"],  4),
            "MAPE (%)"  : round(r.get("MAPE", float("nan")), 2),
            "R²"        : round(r["R2"],    4),
            "Skill"     : round(r["Skill"], 4),
        })
    df = pd.DataFrame(rows).sort_values("MAE (°C)").reset_index(drop=True)
    return df


def save_benchmark_table(
    results: Dict[str, dict],
    path: Optional[str] = None,
) -> pd.DataFrame:
    """Save benchmark table to CSV and return it."""
    df = benchmark_table(results)
    if path is None:
        path = str(
            project_root() / "results" / "tables" / "benchmark_results.csv"
        )
    ensure_dirs(str(Path(path).parent))
    df.to_csv(path, index=False)
    logger.info(f"Benchmark table saved → {path}")
    print("\n" + "="*65)
    print("  Nine-Model Benchmark (Test Set 2018–2025)")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)
    return df


# ─────────────────────────────────────────────────────────────
# Residual diagnostics (paper §8, Figure 9)
# ─────────────────────────────────────────────────────────────

def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "CNN-LSTM",
    save_as: Optional[str] = "fig9_residual_diagnostics.png",
) -> Tuple[plt.Figure, dict]:
    """
    Three-panel residual diagnostic plot (Figure 9 of paper):
        1. Q-Q plot       → tests Gaussian distribution assumption
        2. Residuals vs. Predicted → tests heteroscedasticity
        3. Residual histogram      → shows μ, σ

    Returns figure and summary statistics dict.
    """
    apply_style()
    residuals = y_true - y_pred

    mu    = residuals.mean()
    sigma = residuals.std()
    skew  = stats.skew(residuals)
    kurt  = stats.kurtosis(residuals)
    sw_stat, sw_p = stats.shapiro(residuals[:50])   # Shapiro–Wilk (n≤50)

    logger.info(
        f"[Residuals | {model_name}] "
        f"μ={mu:.4f}°C  σ={sigma:.4f}  skew={skew:.3f}  "
        f"kurt={kurt:.3f}  Shapiro-W p={sw_p:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colours = [PALETTE["cnn_lstm"]] * 3

    # ── Q-Q plot ──────────────────────────────────────────────
    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot of Residuals", fontweight="bold")
    axes[0].get_lines()[0].set(color=colours[0], alpha=0.7, markersize=4)
    axes[0].get_lines()[1].set(color="red", lw=1.5)

    # ── Residuals vs. Predicted ───────────────────────────────
    axes[1].scatter(y_pred, residuals,
                    color=colours[1], alpha=0.5, s=18, edgecolors="none")
    axes[1].axhline(0, color="red", lw=1.5, ls="--")
    axes[1].axhline(mu, color="orange", lw=1.0, ls=":", label=f"μ={mu:.4f}°C")
    axes[1].set_xlabel("Predicted Temperature Anomaly (°C)")
    axes[1].set_ylabel("Residual (°C)")
    axes[1].set_title("Residuals vs. Predicted", fontweight="bold")
    axes[1].legend(fontsize=8)

    # ── Residual histogram ────────────────────────────────────
    axes[2].hist(residuals, bins=40, color=colours[2],
                 alpha=0.75, edgecolor="white", density=True)
    # Overlay Gaussian PDF
    x_pdf = np.linspace(residuals.min(), residuals.max(), 200)
    axes[2].plot(x_pdf, stats.norm.pdf(x_pdf, mu, sigma),
                 "r-", lw=2, label=f"N(μ={mu:.4f}, σ={sigma:.4f})")
    axes[2].axvline(mu, color="red", ls="--", lw=1.2)
    axes[2].set_xlabel("Residual (°C)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Residual Distribution", fontweight="bold")
    axes[2].legend(fontsize=8)

    pct_within_015 = np.mean(np.abs(residuals) <= 0.15) * 100
    fig.suptitle(
        f"Residual Diagnostics — {model_name}   "
        f"[{pct_within_015:.0f}% residuals within ±0.15°C]",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)

    summary = {
        "mean"    : float(mu),
        "std"     : float(sigma),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "shapiro_p": float(sw_p),
        "pct_within_015°C": float(pct_within_015),
    }
    return fig, summary


# ─────────────────────────────────────────────────────────────
# Training curves plot (Figure 8)
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    histories: Dict[str, dict],
    save_as: Optional[str] = "fig7_training_curves.png",
) -> plt.Figure:
    """
    Plot Huber-loss training and validation curves for all DL models.
    Reproduces Figure 8 of the paper.
    """
    apply_style()
    names  = list(histories.keys())
    n      = len(names)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), squeeze=False)
    colour_list = list(PALETTE.values())

    for i, name in enumerate(names):
        ax   = axes[i // ncols][i % ncols]
        hist = histories[name]
        col  = colour_list[i % len(colour_list)]
        ax.semilogy(hist["loss"], color=col, label="Train")
        ax.semilogy(hist["val_loss"], color=col, ls="--", alpha=0.7, label="Validation")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss (log scale)")
        ax.legend(fontsize=8)

    # Turn off unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "Training & Validation Huber-Loss Curves\n"
        "All Deep Learning Architectures",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ─────────────────────────────────────────────────────────────
# Prediction vs. observed plot (Figure 8)
# ─────────────────────────────────────────────────────────────

def plot_predictions(
    results: Dict[str, dict],
    test_dates: pd.DatetimeIndex,
    save_as: Optional[str] = "fig8_test_predictions.png",
) -> plt.Figure:
    """
    Plot observed vs. predicted for all DL models over the test set.
    """
    apply_style()
    names  = [n for n in results if "pred" in results[n]]
    n      = len(names)

    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True, squeeze=False)
    colour_list = list(PALETTE.values())

    for i, name in enumerate(names):
        ax  = axes[i][0]
        res = results[name]
        col = colour_list[i % len(colour_list)]
        ax.plot(test_dates, res["true"], "k-", lw=1.2, alpha=0.85,
                label="Observed (Berkeley Earth)")
        ax.plot(test_dates, res["pred"], color=col, lw=1.5, ls="--",
                label=f"{name} (MAE={res['MAE']:.4f}°C)")
        ax.fill_between(test_dates, res["true"], res["pred"],
                        alpha=0.15, color="orange")
        ax.axhline(1.5, color=PALETTE["paris"], lw=0.8, ls=":",
                   alpha=0.7, label="Paris 1.5°C")
        ax.set_ylabel("Anomaly (°C)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{name}", fontweight="bold")

    axes[-1][0].set_xlabel("Date")
    fig.suptitle(
        "Observed vs. Predicted Global Temperature Anomaly\n"
        "Test Set (Jan 2018 – Dec 2025)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig
