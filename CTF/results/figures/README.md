# results/figures/

All figures from the paper, organised into two sub-sections.

---

## 📁 eda/ — Exploratory Data Analysis figures (§4, Figures 1–6)

| File | Paper Figure | Description |
|------|-------------|-------------|
| `eda_fig0_multimodal_dataset_overview_1880_2024.png` | Fig. 1 | Three-panel multimodal dataset overview 1880–2024: (a) Berkeley Earth temp anomaly, (b) NOAA CO₂, (c) SILSO sunspots |
| `eda_fig1_temperature_anomaly_1960_2024.png` | Fig. 2 | Global surface temperature anomaly 1960–2024 — warm/cool shading + 5-yr moving average |
| `eda_fig2_co2_keeling_curve.png` | Fig. 3a | NOAA Mauna Loa CO₂ — Keeling Curve 315→422 ppm |
| `eda_fig3_sunspot_schwabe_cycles.png` | Fig. 3b | SILSO sunspot number — Cycles 19–25, 11-yr Schwabe smooth |
| `eda_fig4_stl_trend_component.png` | Fig. 4a | STL decomposition — trend component, super-linear acceleration post-2000 |
| `eda_fig5_stl_seasonal_component.png` | Fig. 4b | STL decomposition — seasonal component, stationary ±0.11°C |
| `eda_fig6_feature_correlation_matrix.png` | Fig. 5 | Feature correlation matrix — CO₂ ↔ RF r=1.00; sunspot ↔ temp r=−0.17 to −0.20 |
| `eda_fig7_adf_stationarity_test.png` | Fig. 6a | ADF test: statistic=−2.6249, p=0.0880 → NON-STATIONARY |
| `eda_fig8_autocorrelation_function_acf.png` | Fig. 6b | ACF: significant to lag 48 months, peaks at 12 & 24 → validates 36-month window |

---

## 📁 results figures (§6–§9, Figures 7–13)

| File | Paper Figure | Description |
|------|-------------|-------------|
| `fig07_benchmark_terminal_output.png` | Fig. 7 | Final benchmark results terminal — all 5 DL models: MAE, RMSE, MAPE, R², Skill |
| `fig08a_metric_comparison_mae.png` | Fig. 8a | 9-model MAE comparison bar chart — CNN-LSTM best (0.0944°C) |
| `fig08b_metric_comparison_rmse.png` | Fig. 8b | 9-model RMSE comparison bar chart — CNN-LSTM best (0.1175°C) |
| `fig08c_metric_comparison_r2.png` | Fig. 8c | 9-model R² comparison bar chart — CNN-LSTM best (0.6781) |
| `fig09_observed_vs_predicted_all_models.png` | Fig. 9 | Observed vs. predicted temperature anomaly — all 5 DL models, test set 2018–2025 |
| `fig10_11a_training_curve_lstm.png` | Fig. 10a | LSTM Huber-loss training & validation curves (25 epochs) |
| `fig10_11b_training_curve_bilstm.png` | Fig. 10b | Bi-LSTM Huber-loss training & validation curves (50 epochs) |
| `fig10_11c_training_curve_gru.png` | Fig. 10c | GRU Huber-loss training & validation curves (30 epochs) |
| `fig10_11d_training_curve_transformer.png` | Fig. 10d | Transformer Huber-loss training & validation curves (18 epochs) — note val instability |
| `fig10_11e_training_curve_cnn_lstm.png` | Fig. 10e | CNN-LSTM Huber-loss training & validation curves (28 epochs) — smoothest convergence |
| `fig12a_residuals_vs_predicted.png` | Fig. 11a | CNN-LSTM residuals vs. predicted — no heteroscedasticity |
| `fig12b_qq_plot_residuals.png` | Fig. 11b | CNN-LSTM Q-Q plot — near-Gaussian residuals, tail deviations at El Niño extremes |
| `fig12c_residual_distribution_histogram.png` | Fig. 11c | CNN-LSTM residual histogram — μ=0.0721°C, >80% within ±0.15°C |
| `fig13_decadal_ensemble_forecast_2025_2035.png` | Fig. 12 | 10-year ensemble forecast 2025–2035 — mean converges at 1.25°C below Paris 1.5°C |
