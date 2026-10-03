# LSTM Results

## Frozen method

The LSTM receives the same leakage-safe feature set and target-time partitions as XGBoost.
Feature and target scalers are fitted on training data only.
Validation data controls early stopping and final epoch selection.
The test set is evaluated only after the configuration is frozen.

The architecture is intentionally small: one recurrent layer followed by a regression head.
This keeps the comparison focused on model class rather than extensive neural architecture search.

## Findings

The LSTM did not improve on XGBoost at the primary 15-minute horizon.
It also required more training time and introduced sensitivity to sequence length, scaling, and stopping behaviour.
The result does not imply that recurrent models are unsuitable for all LLM workloads.
It shows that the tested LSTM did not justify its additional complexity for this trace and experimental budget.

The same fixed P95 threshold issue affected burst classification.
When the later-period load distribution shifted downward, positive events became too sparse for stable threshold-based recall and F1.

Detailed training histories, fitted models, scalers, and predictions are generated locally under `outputs/` and `models/`.
They are excluded from Git and can be rebuilt from the frozen configuration.
