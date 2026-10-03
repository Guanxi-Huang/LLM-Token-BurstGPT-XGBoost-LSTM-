# Dataset Sources and Naming

Raw datasets are not committed to this repository.
Download them from the official sources listed in `AzureLLMInferenceDataset2024.md` and place them in this directory.

The local naming convention is `level_dataset_period_source`.
Higher level numbers indicate greater relevance to the main experiment.

- `level_5`: BurstGPT v2.0 `without_fails` traces used for the primary train, validation, test, and robustness experiments.
- `level_4`: complete BurstGPT v2.0 traces used only to audit failed requests with zero response tokens.
- `level_3`: Azure LLM 2024 traces for independent reproduction work.
- `level_2`: Azure LLM 2023 traces for short burst case studies.
- `level_1`: the multimodal Azure trace documented in 2025 and collected from 2024-10-15 to 2024-10-22.

BurstGPT publishes continuous relative timestamps but does not disclose the calendar year of collection.
The filenames therefore use `Undisclosed` instead of an inferred date.

Expected primary files:

```text
level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv
level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv
level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv
```
