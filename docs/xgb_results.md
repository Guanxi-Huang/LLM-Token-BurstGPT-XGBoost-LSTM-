# XGBoost Results

## Frozen method

All lagged and rolling features enter through a one-window shift.
Model and service-type shares are calculated only from completed historical windows.
The 5, 15, and 60-minute targets correspond to 1, 3, and 12 future five-minute steps.
Samples are assigned to splits by target time.

Each horizon uses 12 candidates drawn with seed 42 from a predefined search space.
The selected candidate has the lowest validation MAE, with simpler depth and tree count used only as deterministic tie-breakers.
The test set is not used for feature or parameter selection.

## Selected configurations

| Horizon | Max depth | Learning rate | Trees | Min child weight | Subsample | Column sample |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | 7 | 0.05 | 600 | 5 | 0.9 | 0.7 |
| 15 min | 7 | 0.02 | 600 | 1 | 1.0 | 0.9 |
| 60 min | 7 | 0.05 | 600 | 5 | 0.9 | 0.7 |

## Interpretation

At the primary 15-minute horizon, XGBoost produced the best test MAE in the frozen comparison.
Its advantage was not equally strong in every time segment, and Persistence remained competitive on validation data.
This difference is consistent with temporal distribution shift rather than a universal model ranking.

Historical request volume and recent load statistics were the strongest predictors.
Relative daily and weekly phases added context but did not replace short-term workload history.

The fixed training P95 burst threshold became poorly calibrated after the overall load level fell.
As a result, improved continuous forecast error did not guarantee useful burst F1.
The result supports separate model validation and alert-policy calibration.

Detailed metrics and prediction rows are generated under `outputs/tables/` and are excluded from Git because they are reproducible build artefacts.
