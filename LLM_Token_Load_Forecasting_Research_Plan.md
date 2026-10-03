# Research Plan: Forecasting LLM Token Load and Detecting Traffic Bursts

## Objective

The project evaluates whether recent workload history and request composition can forecast near-term token demand in a real LLM service.
It also tests whether continuous forecasts can support early warning for high-load windows.

## Tasks

The request trace is aggregated into five-minute windows.
For a forecast origin `t`, all features must be available by the end of `t - 5 min`.
The prediction target is the token load at `t + H`, where `H` is 5, 15, or 60 minutes.

The study answers three fixed questions:

1. Which historical features are useful and available at prediction time?
2. Do XGBoost or LSTM improve on Persistence and 24-hour Seasonal Naive baselines?
3. How well do the models detect load above a fixed training-period P95 threshold?

## Data

The primary experiment uses the successful-request traces from BurstGPT v2.0 batches 1 and 2.
Batch 3 is held out from all fitting and selection and is used as a zero-shot cross-period robustness test.
Complete traces that include failed requests are used only for failure-rate audits.

The public timestamps are relative seconds from an undisclosed origin.
Any 1970 UTC timestamp produced by the pipeline is a synthetic resampling index, not a calendar observation.

## Data preparation

1. Validate the schema and numeric fields for each batch.
2. Remove records that are exact duplicates across all public fields.
3. Preserve a full audit of input rows, removed rows, and reasons.
4. Aggregate total tokens, request counts, and request composition into a complete five-minute grid.
5. Retain zero-request windows instead of dropping them.

## Features

Candidate features are limited to information available before the forecast origin:

- lagged token load;
- rolling mean, standard deviation, and maximum after a one-window shift;
- lagged request count and token composition;
- lagged model and log-type shares;
- relative daily and weekly phase encoded with sine and cosine.

Real weekday, weekend, holiday, and absolute calendar features are excluded because the trace does not disclose its real calendar origin.

## Models

- Persistence provides the most recent completed load.
- Seasonal Naive provides the value from the same relative phase 24 hours earlier.
- XGBoost uses a small, predefined hyperparameter search selected by validation MAE.
- LSTM uses a single recurrent layer, frozen feature definitions, and validation-based early stopping.

The model scope is intentionally narrow so that leakage control, baseline quality, and reproducibility receive more attention than architecture count.

## Evaluation

Regression metrics are MAE, RMSE, and sMAPE.
Burst metrics are precision, recall, F1, and average precision.
The burst threshold is calculated once from the training target and then reused unchanged for validation, test, and robustness data.

All splits are chronological by target time.
Scalers, thresholds, and fitted statistics use training data only.
Hyperparameters use validation data only.
The test set is opened only after model selection is frozen.

## Expected contribution

The project provides a compact, reproducible comparison of tree and recurrent models on an operational LLM workload.
Its main contribution is a clear separation between average load forecasting and rare-event alerting under temporal distribution shift.

## Reproducibility requirements

- Random seed: 42.
- Environment: pinned in `requirements.txt`.
- Configuration: stored under `configs/`.
- Data contracts: documented under `docs/`.
- Validation: temporal alignment and output consistency checked by scripts.
- Large data and generated artefacts: excluded from Git and rebuilt locally.
