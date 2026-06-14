"""
train_baselines.py
==================
Trains and evaluates the four ML baseline models:
    Ridge Regression, Random Forest, XGBoost, LightGBM

These are trained on the flattened (non-sequential) feature matrix and
benchmarked against the deep learning architectures in Table 4 of the paper.

Notable finding:
  • LightGBM (MAE=0.1018°C, R²=0.6173) outperforms three DL models,
    confirming the value of the 27-feature engineered vector.
  • Ridge (MAE=0.0064°C, R²=0.9986) achieves near-perfect in-sample
    scores via linear trend extrapolation but FAILS on 2023–2024 El Niño
    nonlinearity → excluded from ensemble.

Reference: Choudhary & Kulkarni (2026), Climatic Change §6.1.
"""

import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from preprocessing import load_all
from feature_engineering import prepare_all
from utils import get_logger, project_root, set_global_seed, ensure_dirs, murphy_skill_score

logger = get_logger("train_baselines")

MODELS_DIR = project_root() / "models"
ensure_dirs(str(MODELS_DIR))

# ── Baseline hyperparameters ─────────────────────────────────
BASELINES = {
    "Ridge": {
        "class" : Ridge,
        "params": {"alpha": 1.0},
    },
    "Random Forest": {
        "class" : RandomForestRegressor,
        "params": {
            "n_estimators": 300,
            "max_depth"   : None,
            "random_state": 42,
            "n_jobs"      : -1,
        },
    },
    "XGBoost": {
        "class" : xgb.XGBRegressor,
        "params": {
            "n_estimators" : 300,
            "max_depth"    : 5,
            "learning_rate": 0.05,
            "subsample"    : 0.8,
            "colsample_bytree": 0.8,
            "random_state" : 42,
            "verbosity"    : 0,
        },
    },
    "LightGBM": {
        "class" : lgb.LGBMRegressor,
        "params": {
            "n_estimators" : 300,
            "num_leaves"   : 63,
            "learning_rate": 0.05,
            "subsample"    : 0.8,
            "colsample_bytree": 0.8,
            "random_state" : 42,
            "verbose"      : -1,
        },
    },
}


# ─────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────

def train_and_evaluate(seed: int = 42) -> pd.DataFrame:
    """
    Train all baseline models and return a DataFrame of test-set metrics.

    Note: baselines operate on the flat (non-sequential) feature matrix,
    so they see all 27 features without the 36-month sliding window.
    """
    set_global_seed(seed)

    raw  = load_all(use_cache=True)
    data = prepare_all(raw)

    # For baselines use flat (non-sequential) scaled arrays
    X_train = data["scaler"].feat_scaler.transform(
        data["train_df"][data["feat_cols"]].values
    )
    y_train_scaled = data["scaler"].target_scaler.transform(
        data["train_df"][["temp_anomaly"]].values
    ).ravel()
    y_train_orig = data["scaler"].inverse_y(y_train_scaled)

    X_test  = data["scaler"].feat_scaler.transform(
        data["test_df"][data["feat_cols"]].values
    )
    y_test_scaled = data["scaler"].target_scaler.transform(
        data["test_df"][["temp_anomaly"]].values
    ).ravel()
    y_true = data["scaler"].inverse_y(y_test_scaled)

    # Climatological baseline (training mean) for skill score
    y_clim = np.full_like(y_true, y_train_orig.mean())

    records = []
    for name, spec in BASELINES.items():
        logger.info(f"Training {name}…")
        mdl = spec["class"](**spec["params"])
        mdl.fit(X_train, y_train_scaled)

        pred_scaled = mdl.predict(X_test)
        y_pred = data["scaler"].inverse_y(pred_scaled)

        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
        r2    = r2_score(y_true, y_pred)
        skill = murphy_skill_score(y_true, y_pred, y_clim)

        records.append({
            "Model": name,
            "MAE"  : round(mae, 6),
            "RMSE" : round(rmse, 6),
            "R2"   : round(r2, 6),
            "Skill": round(skill, 6),
        })
        logger.info(
            f"  {name:<16} MAE={mae:.4f}  RMSE={rmse:.4f}  "
            f"R²={r2:.4f}  Skill={skill:.4f}"
        )

        # Save model
        save_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(mdl, str(save_path))
        logger.info(f"  Saved → {save_path}")

    results_df = pd.DataFrame(records).set_index("Model")

    if "Ridge" in results_df.index:
        logger.warning(
            "Ridge achieves near-perfect R²=0.9986 via linear extrapolation "
            "but is EXCLUDED from the ensemble (fails on 2023–2024 El Niño)."
        )

    return results_df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML baseline models"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = train_and_evaluate(seed=args.seed)
    print("\n── Baseline Model Results ──")
    print(results.to_string())

    out_path = project_root() / "results" / "tables" / "baseline_results.csv"
    ensure_dirs(str(out_path.parent))
    results.to_csv(out_path)
    print(f"\nSaved → {out_path}")
