"""
train_cnn_lstm.py
=================
Trains the CNN-LSTM model — best-performing architecture in the paper
(MAE = 0.0944°C, RMSE = 0.1175°C, R² = 0.6781, Skill = 0.7462).

Architecture (§5.1):
    Conv1D(64, k=3) → Conv1D(32, k=3) → MaxPool(2) → Dropout
    → LSTM(64, return_seq=True) → LSTM(32) → Dense(32) → Dense(1)

Loss : Huber (δ=1.0) — robust to El Niño outliers
Opt  : Adam (lr=1e-3) + ReduceLROnPlateau

Reference: Choudhary & Kulkarni (2026), Climatic Change §5.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from preprocessing import load_all
from feature_engineering import prepare_all
from evaluation import evaluate_single_model
from utils import get_logger, project_root, set_global_seed, ensure_dirs

logger = get_logger("train_cnn_lstm")

# ── Default hyper-parameters (match paper Table 3) ──────────
CFG = {
    "seq_len"         : 36,
    "epochs"          : 120,
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "dropout"         : 0.25,
    "l2_reg"          : 1e-4,
    "patience"        : 15,
    "lr_patience"     : 8,
    "seed"            : 42,
    "model_name"      : "cnn_lstm_best",
}

MODELS_DIR = project_root() / "models"
ensure_dirs(str(MODELS_DIR))


# ─────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────

def build_cnn_lstm(
    seq_len: int,
    n_features: int,
    dropout: float = 0.25,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    CNN-LSTM architecture from the paper.

    The 1D-CNN stage detects local invariant patterns (seasonal cycles,
    ENSO signatures) before the LSTM stage performs global temporal
    integration over the 36-month sequence window.
    """
    model = Sequential(
        [
            Input(shape=(seq_len, n_features)),
            # Local pattern extraction
            Conv1D(64, kernel_size=3, activation="relu", padding="same",
                   kernel_regularizer=l2(l2_reg)),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Dropout(dropout),
            # Global temporal integration
            LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)),
            LSTM(32),
            Dense(32, activation="relu"),
            Dropout(dropout / 2),
            Dense(1),
        ],
        name="CNN-LSTM",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(cfg: dict = CFG) -> dict:
    """
    Run the full CNN-LSTM training pipeline and save the best model.

    Returns
    -------
    dict with keys: model, history, results
    """
    set_global_seed(cfg["seed"])

    # 1. Load and prepare data
    logger.info("Loading and preparing data…")
    raw  = load_all(use_cache=True)
    data = prepare_all(raw, seq_len=cfg["seq_len"])

    X_train, y_train = data["seq_train"]
    X_val,   y_val   = data["seq_val"]
    X_test,  y_test  = data["seq_test"]
    scaler           = data["scaler"]

    n_features = X_train.shape[2]
    logger.info(
        f"Shapes — train: {X_train.shape}  "
        f"val: {X_val.shape}  test: {X_test.shape}  "
        f"features: {n_features}"
    )

    # 2. Build model
    model = build_cnn_lstm(
        seq_len=cfg["seq_len"],
        n_features=n_features,
        dropout=cfg["dropout"],
        l2_reg=cfg["l2_reg"],
        learning_rate=cfg["learning_rate"],
    )
    model.summary(print_fn=logger.info)

    # 3. Callbacks
    ckpt_path = MODELS_DIR / f"{cfg['model_name']}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg["lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # 4. Train
    logger.info("Training CNN-LSTM…")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(
        f"Training complete — best val_loss: "
        f"{min(history.history['val_loss']):.5f}  "
        f"({len(history.history['loss'])} epochs)"
    )

    # 5. Evaluate on test set
    results = evaluate_single_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        name="CNN-LSTM",
    )

    # 6. Save model
    model.save(str(ckpt_path))
    logger.info(f"Model saved → {ckpt_path}")

    return {
        "model"  : model,
        "history": history.history,
        "results": results,
        "data"   : data,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN-LSTM for climate forecasting")
    p.add_argument("--seq-len",  type=int,   default=36,    help="Lookback window (months)")
    p.add_argument("--epochs",   type=int,   default=120,   help="Max training epochs")
    p.add_argument("--lr",       type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--batch",    type=int,   default=32,    help="Batch size")
    p.add_argument("--dropout",  type=float, default=0.25,  help="Dropout rate")
    p.add_argument("--seed",     type=int,   default=42,    help="Random seed")
    p.add_argument("--no-cache", action="store_true",       help="Re-download data")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(CFG)
    cfg.update({
        "seq_len"       : args.seq_len,
        "epochs"        : args.epochs,
        "learning_rate" : args.lr,
        "batch_size"    : args.batch,
        "dropout"       : args.dropout,
        "seed"          : args.seed,
    })
    out = train(cfg)
    r = out["results"]
    print(
        f"\n{'='*55}\n"
        f"  CNN-LSTM Test Results\n"
        f"{'='*55}\n"
        f"  MAE   = {r['MAE']:.4f} °C\n"
        f"  RMSE  = {r['RMSE']:.4f} °C\n"
        f"  MAPE  = {r['MAPE']:.2f} %\n"
        f"  R²    = {r['R2']:.4f}\n"
        f"  Skill = {r['Skill']:.4f}\n"
        f"{'='*55}"
    )
