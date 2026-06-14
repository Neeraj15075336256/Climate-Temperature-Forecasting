# 🌍 Climate-Temperature-Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20686696.svg)](https://doi.org/10.5281/zenodo.20686696)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neeraj15075336256/Climate-Temperature-Forecasting/blob/main/notebooks/03_Model_Training.ipynb)

> **Global Temperature Forecasting Using Multimodal Machine Learning: An Empirical Study Integrating CO₂ and Solar Activity Data**  
> Neeraj Choudhary · Sheetal Abhijit Kulkarni  
> Department of Computer Engineering and Technology, Dr. Vishwanath Karad MIT–World Peace University, Pune, India  
> *Submitted to Climatic Change (Springer), 2026*

---

## 📖 Abstract

A reproducible multimodal machine learning framework for global surface temperature forecasting integrating **144 years (1880–2024)** of:

- 🌡️ **Berkeley Earth** — global land/ocean temperature anomalies  
- 🏭 **NOAA Mauna Loa** — atmospheric CO₂ concentrations (Keeling Curve)  
- ☀️ **SILSO** — international sunspot numbers (Schwabe cycles)

**Five deep learning architectures** (LSTM, Bi-LSTM, GRU, Transformer, CNN-LSTM) trained with Huber loss are benchmarked against four ML baselines in a **nine-model evaluation framework**.

---

## 📊 Results Summary (Test Set: 2018–2025)

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | R² | Skill Score |
|-------|----------|-----------|----------|----|-------------|
| **CNN-LSTM ★** | **0.0944** | **0.1175** | **6.98** | **0.6781** | **0.7462** |
| GRU | 0.1074 | 0.1383 | 7.87 | 0.5538 | 0.7013 |
| Transformer | 0.1146 | 0.1453 | 8.93 | 0.5078 | 0.6862 |
| Bi-LSTM | 0.1313 | 0.1557 | 9.61 | 0.4344 | 0.6636 |
| LSTM | 0.1335 | 0.1586 | 9.75 | 0.4135 | 0.6575 |
| LightGBM | 0.1018 | 0.1317 | — | 0.6173 | — |
| Random Forest | 0.1216 | 0.1522 | — | 0.4884 | — |
| XGBoost | 0.1262 | 0.1615 | — | 0.4243 | — |
| Ridge* | 0.0064* | 0.0079* | — | 0.9986* | — |

*\*Ridge excluded from ensemble — linear trend extrapolation artefact; fails on 2023–2024 El Niño nonlinearity.*

**Decadal ensemble mean forecast (2025–2035): 1.25°C — below the Paris Agreement 1.5°C threshold.**

---

## 🗂️ Repository Structure

```
Climate-Temperature-Forecasting/
│
├── README.md               ← This file
├── LICENSE                 ← MIT License + dataset licences
├── requirements.txt        ← pip dependencies (pinned)
├── environment.yml         ← conda environment
├── CITATION.cff            ← Machine-readable citation
├── .gitignore
├── .gitattributes          ← Git LFS config for .keras/.pkl/.png
│
├── data/
│   ├── raw/
│   │   ├── berkeley_earth_temperature.csv   ← Temperature anomaly 1880–2024
│   │   ├── noaa_co2_mlo.csv                 ← Mauna Loa CO₂ 1958–2024
│   │   └── silso_sunspots.csv               ← Sunspot number 1749–2024
│   └── processed/
│       ├── climate_merged_features.csv      ← 27-feature matrix (1,740 rows)
│       └── train_val_test_split.csv         ← Split metadata
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb          ← §3.1: data acquisition & cleaning
│   ├── 02_Exploratory_Data_Analysis.ipynb   ← §4: 8-panel EDA (STL, ADF, ACF)
│   ├── 03_Model_Training.ipynb              ← §5: all 5 DL + 4 ML baselines
│   └── 04_Model_Evaluation.ipynb            ← §6–9: metrics, residuals, forecast
│
├── src/
│   ├── preprocessing.py        ← Data loading, cleaning, merging
│   ├── feature_engineering.py  ← 27-feature construction + ClimateScaler
│   ├── train_cnn_lstm.py       ← CNN-LSTM (best model, MAE=0.0944°C)
│   ├── train_transformer.py    ← Transformer architecture
│   ├── train_baselines.py      ← Ridge, RF, XGBoost, LightGBM
│   ├── ensemble_model.py       ← 4-model ensemble + decadal projection
│   ├── prediction.py           ← Rolling multi-step forecast + CO₂/SSN extrapolation
│   ├── evaluation.py           ← MAE/RMSE/MAPE/R²/Murphy Skill + residual diagnostics
│   └── utils.py                ← Logging, seeding, plotting helpers
│
├── models/                     ← Trained weights (Git LFS / Zenodo)
│   ├── cnn_lstm_best.keras
│   ├── gru_best.keras
│   ├── lstm_best.keras
│   ├── bilstm_best.keras
│   ├── transformer_best.keras
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── random_forest_model.pkl
│   └── README.md
│
├── results/
│   ├── figures/
│   │   ├── README.md                        ← Full figure index
│   │   ├── eda/                             ← §4 EDA figures (Figs 1–6)
│   │   │   ├── eda_fig0_multimodal_dataset_overview_1880_2024.png
│   │   │   ├── eda_fig1_temperature_anomaly_1960_2024.png
│   │   │   ├── eda_fig2_co2_keeling_curve.png
│   │   │   ├── eda_fig3_sunspot_schwabe_cycles.png
│   │   │   ├── eda_fig4_stl_trend_component.png
│   │   │   ├── eda_fig5_stl_seasonal_component.png
│   │   │   ├── eda_fig6_feature_correlation_matrix.png
│   │   │   ├── eda_fig7_adf_stationarity_test.png
│   │   │   └── eda_fig8_autocorrelation_function_acf.png
│   │   ├── fig07_benchmark_terminal_output.png
│   │   ├── fig08a_metric_comparison_mae.png
│   │   ├── fig08b_metric_comparison_rmse.png
│   │   ├── fig08c_metric_comparison_r2.png
│   │   ├── fig09_observed_vs_predicted_all_models.png
│   │   ├── fig10_11a_training_curve_lstm.png
│   │   ├── fig10_11b_training_curve_bilstm.png
│   │   ├── fig10_11c_training_curve_gru.png
│   │   ├── fig10_11d_training_curve_transformer.png
│   │   ├── fig10_11e_training_curve_cnn_lstm.png
│   │   ├── fig12a_residuals_vs_predicted.png
│   │   ├── fig12b_qq_plot_residuals.png
│   │   ├── fig12c_residual_distribution_histogram.png
│   │   └── fig13_decadal_ensemble_forecast_2025_2035.png
│   └── tables/
│       ├── table1_literature_review.csv     ← 25-row literature review
│       ├── table2_dataset_summary.csv       ← Dataset specifications
│       ├── table3_hyperparameter_config.csv ← Unified training config
│       └── table4_benchmark_results.csv     ← 9-model benchmark (Table 4)
│
├── docs/
│   ├── manuscript.pdf                       ← Added post-acceptance
│   ├── supplementary_material.pdf           ← Added post-acceptance
│   └── README.md
│
└── zenodo/
    └── metadata.json                        ← Zenodo upload metadata
```

---

## 🚀 Quick Start

### Option 1 — Google Colab *(recommended, zero setup)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neeraj15075336256/Climate-Temperature-Forecasting/blob/main/notebooks/03_Model_Training.ipynb)

All three datasets are fetched automatically from public APIs. No downloads required.

### Option 2 — Local conda environment

```bash
git clone https://github.com/Neeraj15075336256/Climate-Temperature-Forecasting.git
cd Climate-Temperature-Forecasting
conda env create -f environment.yml
conda activate climate-forecasting
python src/train_cnn_lstm.py        # Train best model
python src/evaluation.py            # Evaluate all models
python src/ensemble_model.py        # 10-year decadal forecast
```

### Option 3 — pip

```bash
pip install -r requirements.txt
jupyter notebook notebooks/03_Model_Training.ipynb
```

---

## 📋 Reproducing Paper Results

Run notebooks in order:

| Step | Notebook | Paper Section | Key Output |
|------|----------|--------------|------------|
| 1 | `01_Data_Preprocessing.ipynb` | §3.1 | `data/processed/climate_merged_features.csv` |
| 2 | `02_Exploratory_Data_Analysis.ipynb` | §4 | `results/figures/eda/` (9 EDA figures) |
| 3 | `03_Model_Training.ipynb` | §5 | `models/*.keras`, training curves |
| 4 | `04_Model_Evaluation.ipynb` | §6–9 | Table 4, Figs 8–13, decadal forecast |

> **Seed**: `42` everywhere. Expected CNN-LSTM test MAE: **0.0944°C ± 0.003°C** (minor GPU non-determinism).

---

## 🔬 Key Contributions

| # | Contribution | Paper Section |
|---|-------------|--------------|
| 1 | 8-panel EDA: STL decomposition, ADF stationarity (p=0.088), ACF persistence to 48 months | §4 |
| 2 | 27-feature multimodal engineering: CO₂ radiative forcing RF=5.35×ln(C/C₀), Schwabe cycles, lag/MA features | §3.2 |
| 3 | Unified 9-model benchmark with Murphy Skill Score — CNN-LSTM best across all 5 metrics | §6 |
| 4 | Huber-loss training dynamics analysis across all architectures | §7 |
| 5 | CNN-LSTM residual diagnostics validating Gaussian error assumption (μ=0.0721°C) | §8 |
| 6 | 10-year ensemble projection 2025–2035 — mean 1.25°C, below Paris 1.5°C threshold | §9 |

---

## 📁 Data Sources

| Dataset | Period | Source | Reference |
|---------|--------|--------|-----------|
| Berkeley Earth Temperature | 1880–2024 | [berkeleyearth.org](https://berkeleyearth.org/data) | Rohde & Hausfather (2020) |
| NOAA Mauna Loa CO₂ | 1958–2024 | [gml.noaa.gov](https://gml.noaa.gov/ccgg/trends/) | NOAA GML (2025) |
| SILSO Sunspot No. V2.0 | 1749–2024 | [sidc.be](https://www.sidc.be/silso/datafiles) | Clette & Lefèvre (2015) |

---

## 📄 Citation

If you use this code or data, please cite:

```bibtex
@article{choudhary2026global,
  title   = {Global Temperature Forecasting Using Multimodal Machine Learning:
             An Empirical Study Integrating {CO}$_2$ and Solar Activity Data},
  author  = {Choudhary, Neeraj and Kulkarni, Sheetal Abhijit},
  journal = {Climatic Change},
  year    = {2026},
  doi     = {10.XXXX/climatic-change.XXXX},
  note    = {Under review}
}
```

Software archive:

```bibtex
@software{choudhary2026software,
  author    = {Choudhary, Neeraj and Kulkarni, Sheetal Abhijit},
  title     = {Climate-Temperature-Forecasting: A Reproducible Multimodal
               Machine Learning Pipeline for Global Surface Temperature Prediction},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20686696},
  url       = {https://doi.org/10.5281/zenodo.20686696}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20686696.svg)](https://doi.org/10.5281/zenodo.20686696)

---

## 👤 Authors

| Author | Role | ORCID | Contact |
|--------|------|-------|---------|
| **Neeraj Choudhary** | First author, implementation | [0009-0008-9470-0281](https://orcid.org/0009-0008-9470-0281) | neeraj.choudhary@mitwpu.edu.in |
| **Dr. Sheetal Abhijit Kulkarni** | Supervisor, corresponding author | — | sheetal.kulkarni@mitwpu.edu.in |

Dr. Vishwanath Karad MIT–World Peace University, Pune, India

---

## 📜 License

**Code**: MIT License — see [LICENSE](LICENSE)  
**Data**: Berkeley Earth (CC BY 4.0) · NOAA GML (Public Domain) · SILSO (CC BY-NC 4.0)

---

## 🙏 Acknowledgements

The authors thank **Berkeley Earth**, **NOAA Global Monitoring Laboratory**, and **SILSO World Data Center** (Royal Observatory of Belgium) for open access to climate data. Computational experiments were performed on **Google Colab**.
