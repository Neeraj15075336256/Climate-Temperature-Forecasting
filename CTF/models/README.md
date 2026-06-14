# Trained Model Weights

This directory stores trained model weights for the **Global Temperature
Forecasting via a Reproducible Multimodal Machine Learning Pipeline**
project.

## Files (generated after running the training script)

| File | Model | Type |
|------|-------|------|
| `cnn_lstm_best.keras` | CNN-LSTM | Deep learning |
| `gru_best.keras` | GRU | Deep learning |
| `lstm_best.keras` | LSTM | Deep learning |
| `bilstm_best.keras` | Bi-LSTM | Deep learning |
| `transformer_best.keras` | Transformer | Deep learning |
| `lightgbm_model.pkl` | LightGBM | Tree ensemble |
| `xgboost_model.pkl` | XGBoost | Tree ensemble |
| `random_forest_model.pkl` | Random Forest | Tree ensemble |

> Exact MAE/RMSE values depend on the data fetched at training time
> (live data from NOAA / Berkeley Earth / SILSO, or the synthetic
> fallback if those sources are unreachable). See
> `models/benchmark_results.csv` after a run for current metrics.

## How to generate

All 8 models are trained and saved by a single self-contained script:

```bash
pip install -r requirements.txt
python generate_models.py
```

This will:
1. Fetch (or synthesize, if offline) Berkeley Earth temperature, NOAA
   CO₂, and SILSO sunspot data.
2. Run feature engineering and preprocessing.
3. Train all 5 deep learning models (LSTM, Bi-LSTM, GRU, Transformer,
   CNN-LSTM) and 3 tree-based baselines (Random Forest, XGBoost,
   LightGBM).
4. Save all 8 model files into `models/`.

## Loading the models

```python
import joblib
from tensorflow.keras.models import load_model

lstm = load_model("models/lstm_best.keras")
rf   = joblib.load("models/random_forest_model.pkl")
```

## Pre-trained weights

If you'd rather not retrain, pre-trained weights can be archived on
Zenodo or another release host and downloaded directly into `models/`.
Update this section with your archive link/DOI once published.

## Git LFS

`.keras` and `.pkl` files in this directory are typically large
(several MB to tens of MB). If tracking them in Git, use Git LFS:

```bash
git lfs track "models/*.keras" "models/*.pkl"
git lfs pull
```
