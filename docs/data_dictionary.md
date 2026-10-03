# BurstGPT v2.0 Data Dictionary and Batch Rules

## Time semantics

`Timestamp` is relative seconds from the beginning of each released trace.
The pipeline converts it to a synthetic UTC index anchored at `1970-01-01` only so Pandas can resample the data.
The synthetic date is not the real collection date and must not be used to infer weekday, weekend, holiday, or season.

## Request fields

| Concept | Source field | Use |
| --- | --- | --- |
| Relative request time | `Timestamp` | Ordering and five-minute aggregation |
| Model family | `Model` | Historical request composition |
| Input length | `Request tokens` | Token load and historical composition |
| Output length | `Response tokens` | Token load and failure audit |
| Total request size | `Total tokens` | Checked against input plus output |
| Service category | `Log Type` | Historical API and conversation shares |
| Session identifier | `Session ID` | Batch 3 audit only |
| Request duration | `Elapsed time` | Audit only; excluded because it is known after completion |

The main target is the sum of request and response tokens in each five-minute window.

## Batch schema

Batches 1 and 2 contain:

```text
Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type
```

Batch 3 also contains `Session ID` and `Elapsed time`.
The additional fields are not used as model inputs because they are unavailable or inconsistent across the primary batches.

## Cleaning rules

The cleaner reports all removed rows and their reasons.
It checks timestamp validity, numeric token fields, negative values, missing required fields, total-token consistency, and exact duplicates.
Exact duplicates are defined across all public fields in a batch.

The public data does not contain a unique request identifier.
Two legitimate concurrent requests may therefore look identical, so duplicate removal is retained as an explicit study limitation.

## Aggregation contract

- Frequency: five minutes.
- Interval: left-closed and right-open.
- Missing windows: inserted into the complete grid.
- Empty-window load and counts: zero.
- Empty-window means and shares: missing unless explicitly defined by downstream code.
- Forecast features: shifted so only completed windows are available.

## Experimental roles

| Data | Role |
| --- | --- |
| `without_fails_1` | Primary chronological train, validation, and test |
| `without_fails_2` | Primary chronological train, validation, and test |
| `without_fails_3` | Zero-shot cross-period robustness test only |
| `complete_1` | Failed-request audit only |

No batch 3 record is used to fit a scaler, model, threshold, or hyperparameter.
