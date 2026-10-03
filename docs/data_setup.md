# Data and Directory Setup

## Required raw files

Place these files in `Dataset/`:

```text
level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv
level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv
level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv
```

Batches 1 and 2 form the primary experiment.
Batch 3 is reserved for the cross-period robustness test.

## Generated directories

- `data/processed/` contains cleaned requests, time series, and feature partitions.
- `models/` contains fitted models and scalers.
- `outputs/tables/` contains metrics, audits, and aligned predictions.
- `outputs/figures/` contains generated figures.

These directories are local build outputs and are excluded from Git.

## Naming conventions

- Figure: `fig_01_daily_load.png`
- Table: `table_01_baseline_test.csv`
- XGBoost model: `xgb_h15.joblib`
- LSTM model: `lstm_h15.keras`
- Feature split: `features_h15_train.parquet`

## Rebuild rule

Run scripts from the repository root.
Do not edit generated tables or figures manually.
If a stage fails, fix the source or configuration and rebuild the affected outputs.
