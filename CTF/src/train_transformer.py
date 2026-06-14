"""
train_transformer.py
====================
Trains the Transformer model (paper §5.1).

Architecture:
    Dense(64) positional projection
    → 2 × Transformer blocks (4-head attention, key_dim=16, ff_dim=128)
    → GlobalAveragePooling1D → Dense(64) → Dense(1)

Loss: Huber (δ=1.0), lr=5e-4 (half of other models — Transformer is
more sensitive to learning rate).

NOTE: The Transformer diverges to ≈0.94°C post-2027 in decadal
projection due to attention-weight extrapolation instability on the
2023–2024 El Niño. It is excluded from the ensemble mean. See §9.

Reference: Choudhary & Kulkarni (2026), Climatic Change §5.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add, Dense, Dropout, GlobalAveragePooling1D, Input,
    LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from preprocessing import load_all
from feature_engineering import prepare_all
from evaluation import evaluate_single_model
from utils import get_logger, project_root, set_global_seed, ensure_dirs

logger = get_logger("train_transformer")

CFG = {
    "seq_len"              : 36,
    "n_heads"              : 4,
    "key_dim"              : 16,
    "ff_dim"               : 128,
    "n_transformer_blocks" : 2,
    "proj_dim"             : 64,
    "dropout"              : 0.25,
    "epochs"               : 120,
    "batch_size"           : 32,
    "learning_rate"        : 5e-4,   # lower than RNN models
    "patience"             : 15,
    "lr_patience"          : 8,
    "seed"                 : 42,
    "model_name"           : "transformer_best",
}

MODELS_DIR = project_root() / "models"
ensure_dirs(str(MODELS_DIR))


# ─────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────

def _transformer_block(
    x: tf.Tensor,
    n_heads: int,
    key_dim: int,
    ff_dim: int,
    dropout: float,
) -> tf.Tensor:
    """Single Transformer encoder block with pre-LN residual connections."""
    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))
    # Feed-forward
    ff  = Dense(ff_dim, activation="gelu")(x)
    ff  = Dense(x.shape[-1])(ff)
    ff  = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(Add()([x, ff]))


def build_transformer(
    seq_len: int,
    n_features: int,
    n_heads: int   = 4,
    key_dim: int   = 16,
    ff_dim: int    = 128,
    n_blocks: int  = 2,
    proj_dim: int  = 64,
    dropout: float = 0.25,
    lr: float      = 5e-4,
) -> Model:
    """
    Transformer encoder for climate time-series forecasting.

    Sinusoidal positional encoding is implicitly captured via the
    Dense projection of each time step.
    """
    inp = Input(shape=(seq_len, n_features))
    x   = Dense(proj_dim)(inp)  # positional projection

    for _ in range(n_blocks):
        x = _transformer_block(x, n_heads, key_dim, ff_dim, dropout)

    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model(inp, out, name="Transformer")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(cfg: dict = CFG) -> dict:
    set_global_seed(cfg["seed"])

    raw  = load_all(use_cache=True)
    data = prepare_all(raw, seq_len=cfg["seq_len"])
    X_train, y_train = data["seq_train"]
    X_val,   y_val   = data["seq_val"]
    X_test,  y_test  = data["seq_test"]
    scaler            = data["scaler"]

    n_features = X_train.shape[2]
    model = build_transformer(
        seq_len=cfg["seq_len"],
        n_features=n_features,
        n_heads=cfg["n_heads"],
        key_dim=cfg["key_dim"],
        ff_dim=cfg["ff_dim"],
        n_blocks=cfg["n_transformer_blocks"],
        proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"],
        lr=cfg["learning_rate"],
    )

    ckpt_path = MODELS_DIR / f"{cfg['model_name']}.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=cfg["lr_patience"], min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    logger.info("Training Transformer…")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    results = evaluate_single_model(model, X_test, y_test, scaler, "Transformer")
    model.save(str(ckpt_path))
    logger.info(f"Transformer saved → {ckpt_path}")
    logger.warning(
        "NOTE: Transformer is excluded from the decadal ensemble mean "
        "due to attention-weight extrapolation instability (see §9 of paper)."
    )
    return {"model": model, "history": history.history,
            "results": results, "data": data}


if __name__ == "__main__":
    out = train()
    r = out["results"]
    print(
        f"\n  Transformer Test Results\n"
        f"  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
        f"R²={r['R2']:.4f}  Skill={r['Skill']:.4f}"
    )
