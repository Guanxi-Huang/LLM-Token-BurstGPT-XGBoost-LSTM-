# Week 5 Python + RStudio visualization catalog

The two toolchains read the same reviewed files in `outputs/tables/w5_visualization/`. Python figures are written to `outputs/figures/w5_python/`; RStudio figures are written to `outputs/figures/w5_rstudio/`. Every chart uses Token per 5-minute target window, the frozen train-only P95, the same model order, and the same color mapping.

| ID | Analytical question | Python file | RStudio file | Interpretation boundary |
|---|---|---|---|---|
| W5-01 | Which model has the lowest h15 MAE/RMSE, with uncertainty? | `fig_w5_py_01_h15_regression_accuracy.png` | `fig_w5_r_01_h15_regression_accuracy.png` | Test estimates are reporting-only; overlapping intervals caution against overclaiming small gaps. |
| W5-02 | Do overall MAE and burst F1 rankings persist at 5/15/60 minutes? | `fig_w5_py_02_mae_by_horizon.png` | `fig_w5_r_02_mae_by_horizon.png` | The 15-minute task is primary; 5/60-minute results are appendix evidence. |
| W5-03 | How do forecasts track an objectively selected typical test week? | `fig_w5_py_03_representative_week_h15.png` | `fig_w5_r_03_representative_week_h15.png` | The week is selected by a deterministic median-load rule, not visual preference. |
| W5-04 | What happens around the maximum observed test-load window? | `fig_w5_py_04_burst_zoom_h15.png` | `fig_w5_r_04_burst_zoom_h15.png` | The train P95 is fixed; the zoom explains threshold hits and misses. |
| W5-05 | How well do forecast scores rank rare burst windows? | `fig_w5_py_05_pr_curves_h15.png` | `fig_w5_r_05_pr_curves_h15.png` | AP is ranking evidence and does not replace the frozen operating-point F1. |
| W5-06 | Which errors are TP/FP/FN/TN at the fixed P95? | `fig_w5_py_06_confusion_matrices_h15.png` | `fig_w5_r_06_confusion_matrices_h15.png` | Counts and row shares prevent ordinary windows from hiding rare-event misses. |
| W5-07 | How do MAE and signed bias differ between burst and non-burst windows? | `fig_w5_py_07_burst_segment_mae_bias_h15.png` | `fig_w5_r_07_burst_segment_mae_bias_h15.png` | Overall error and extreme-load error are reported as separate findings. |
| W5-08 | Which pre-specified phase/composition groups have higher MAE or bias? | `fig_w5_py_08_segment_profiles_h15.png` | `fig_w5_r_08_segment_profiles_h15.png` | Relative week phase is not a real weekday; target-time GPT-4/API shares are diagnostic only. |
| W5-09 | What does the minimal h15 XGBoost ablation show on validation and test? | `fig_w5_py_09_xgb_ablation_h15.png` | `fig_w5_r_09_xgb_ablation_h15.png` | Feature selection uses validation MAE only, even when test ranking differs. |
| W5-10 | Do overall error and burst F1 transfer to BurstGPT-3? | `fig_w5_py_10_burstgpt3_robustness.png` | `fig_w5_r_10_burstgpt3_robustness.png` | This is an external zero-shot cross-period test; batch 3 is never used for fitting or selection. |

Reproduction order:

```powershell
python src/07_evaluate.py --bootstrap-replicates 1000
python src/11_week5_experiments.py --bootstrap-replicates 500
python src/12_visualize_w5_python.py
E:\R\R-4.5.2\bin\Rscript.exe notebooks/04_w5_visualization_rstudio.R
python scripts/write_w5_dual_visual_results_to_docx.py
```
