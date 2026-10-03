# Forecasting LLM Token Load and Detecting Traffic Bursts

This project studies short-term workload forecasting for a real large language model service using the BurstGPT v2.0 request traces.
It converts request-level records into complete five-minute windows and compares Persistence, Seasonal Naive, XGBoost, and a single-layer LSTM at 5, 15, and 60-minute horizons.
The main evaluation uses a 15-minute horizon, while the other horizons are reported as supporting evidence.

## Research questions

1. Which temporal and request-composition features describe short-term token load without using future information?
2. How do XGBoost and LSTM compare with simple operational baselines under a strict time split?
3. Can the same forecasts identify capacity bursts defined by a threshold fixed on the training period?

## Methodology

- Target: total request and response tokens per five-minute window.
- Window convention: left-closed and right-open intervals, `[t, t + 5 min)`.
- Forecast horizons: 5, 15, and 60 minutes.
- Split: chronological 70% training, 15% validation, and 15% test by target time.
- Burst label: `token_load >= P95(training token_load)`.
- Selection: hyperparameters and stopping decisions use validation data only.
- Robustness: BurstGPT batch 3 is used only as a zero-shot cross-period test.
- Reproducibility: all random procedures use seed 42.

## Main findings

- At the 15-minute horizon, XGBoost achieved the lowest test MAE among the evaluated methods.
- LSTM did not improve on XGBoost for this dataset and experimental budget.
- Persistence remained a strong validation baseline, showing that short-term load continuity matters.
- The fixed training-period P95 threshold produced very few positive events after a large distribution shift.
- Better average forecast error did not automatically produce reliable burst alerts.

The central engineering result is that forecasting quality and alert calibration must be evaluated separately.
A model can reduce average capacity error while still missing rare threshold crossings under temporal drift.

## Repository structure

```text
configs/       Frozen model and experiment configuration
Dataset/       Dataset metadata; raw traces are excluded from Git
docs/          Data contracts, setup notes, and model results
scripts/       Reproducible build and validation entry points
src/           Cleaning, feature engineering, modelling, and evaluation
requirements.* Python dependency definitions
```

Generated data, trained models, figures, document-editing workspaces, and local process records are intentionally excluded through `.gitignore`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the public traces described in [Dataset/README.md](Dataset/README.md), place them in `Dataset/`, and then run the pipeline from the repository root.

## Reproduce the core pipeline

```powershell
python src/01_clean.py
python src/02_build_series.py
python src/03_features.py
python src/04_baselines.py
python src/05_xgboost.py --horizons-minutes 5 15 60 --n-iter 12 --shap-sample-size 2000
python src/06_lstm.py --horizons-minutes 5 15 60
python src/07_evaluate.py
```

Run the validation scripts after the corresponding stage.
The scripts fail loudly on temporal leakage, alignment errors, missing artefacts, or inconsistent metrics.

## Documentation

- [Research plan](LLM_Token_Load_Forecasting_Research_Plan.md)
- [Data setup](docs/data_setup.md)
- [Data dictionary](docs/data_dictionary.md)
- [XGBoost results](docs/xgb_results.md)
- [LSTM results](docs/lstm_results.md)

Chinese and bilingual working documents are preserved locally with language suffixes in their filenames and are excluded from Git.

## Scope and limitations

- Public timestamps are relative, so the synthetic 1970 UTC index is used only for resampling.
- The trace has no unique request identifier, which makes exact duplicate removal a documented assumption.
- Results describe the released BurstGPT periods and should not be treated as universal production guarantees.
- Raw data and generated artefacts are not redistributed in this repository.
