# Week-5 visualization bundle for RStudio.
#
# Source this file in RStudio or run it with Rscript from the project root.
# It reads only the shared CSVs written by src/12_visualize_w5_python.py, so the
# Python and RStudio figures use identical frozen Week-5 observations.

Sys.unsetenv(c("LC_ALL", "LC_CTYPE", "LC_COLLATE", "LC_MONETARY", "LC_TIME"))
try(suppressWarnings(Sys.setlocale("LC_ALL", "Chinese (Simplified)_China.utf8")), silent = TRUE)
options(encoding = "UTF-8", scipen = 999)
suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(scales)
  library(grid)
})

find_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/"))
  candidate <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(candidate)) return(normalizePath(candidate, winslash = "/"))
  NULL
}

script_path <- find_script_path()
root <- if (!is.null(script_path)) dirname(dirname(script_path)) else normalizePath(getwd(), winslash = "/")
shared <- Sys.getenv("W5_SHARED_DIR", unset = file.path(root, "outputs", "tables", "w5_visualization"))
figures <- Sys.getenv("W5_FIGURES_DIR", unset = file.path(root, "outputs", "figures", "w5_rstudio"))
if (!dir.exists(shared)) stop("Run src/12_visualize_w5_python.py before the RStudio bundle")
dir.create(figures, recursive = TRUE, showWarnings = FALSE)

INK <- "#232A31"
MUTED <- "#5B6470"
GRID <- "#D9DEE5"
OPEN <- "#EFF3F7"
model_order <- c("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
model_colors <- c("Persistence" = "#31688E", "Seasonal Naive" = "#E69F00", "XGBoost" = "#D55E00", "LSTM" = "#7A8F2A")
model_linetypes <- c("Persistence" = "dashed", "Seasonal Naive" = "dotdash", "XGBoost" = "solid", "LSTM" = "dotted")

theme_w5 <- function(base_size = 10) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = GRID, linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.line.x = element_line(color = INK, linewidth = 0.35),
      axis.line.y = element_line(color = INK, linewidth = 0.35),
      plot.title = element_text(face = "bold", size = rel(1.32), color = INK, hjust = 0),
      plot.subtitle = element_text(size = rel(0.86), color = MUTED, margin = margin(b = 8)),
      plot.caption = element_text(size = rel(0.70), color = MUTED, hjust = 0, margin = margin(t = 8)),
      axis.title = element_text(color = INK),
      axis.text = element_text(color = INK),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      strip.text = element_text(face = "bold", color = INK),
      strip.background = element_rect(fill = OPEN, color = NA)
    )
}

save_plot <- function(plot, filename, width = 11.5, height = 6.0) {
  path <- file.path(figures, filename)
  ggsave(path, plot = plot, width = width, height = height, dpi = 220, bg = "white")
  path
}

metrics <- read_csv(file.path(shared, "w5_metrics_all.csv"), show_col_types = FALSE)
week <- read_csv(file.path(shared, "w5_representative_week_h15.csv"), show_col_types = FALSE)
zoom <- read_csv(file.path(shared, "w5_burst_zoom_h15.csv"), show_col_types = FALSE)
pr_curve <- read_csv(file.path(shared, "w5_pr_curve_h15.csv"), show_col_types = FALSE)
segment <- read_csv(file.path(shared, "w5_segment_errors_h15.csv"), show_col_types = FALSE)
ablation <- read_csv(file.path(shared, "w5_ablation_h15.csv"), show_col_types = FALSE)
robustness <- read_csv(file.path(shared, "w5_robustness_all.csv"), show_col_types = FALSE)
metadata <- read_csv(file.path(shared, "w5_visualization_metadata.csv"), show_col_types = FALSE)

metrics$model <- factor(metrics$model, levels = model_order)
week$model <- factor(week$model, levels = model_order)
zoom$model <- factor(zoom$model, levels = model_order)
pr_curve$model <- factor(pr_curve$model, levels = model_order)
segment$model <- factor(segment$model, levels = model_order)
robustness$model <- factor(robustness$model, levels = model_order)

# W5-R-01: primary h15 MAE/RMSE with confidence intervals.
main <- metrics[metrics$horizon_minutes == 15, ]
reg_long <- rbind(
  data.frame(model = main$model, metric = "MAE", estimate = main$mae, low = main$mae_ci_low, high = main$mae_ci_high),
  data.frame(model = main$model, metric = "RMSE", estimate = main$rmse, low = main$rmse_ci_low, high = main$rmse_ci_high)
)
reg_long$metric <- factor(reg_long$metric, levels = c("MAE", "RMSE"))
p1 <- ggplot(reg_long, aes(x = model, y = estimate, fill = model)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.3, show.legend = FALSE) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.18, color = INK, linewidth = 0.45) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()), expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "W5-R-01 Main 15-minute regression accuracy",
    subtitle = "Frozen test estimates with 95% synthetic-day block-bootstrap intervals",
    x = NULL, y = "Token per 5-minute target window",
    caption = "Source: table_04_main_results.csv | reporting only; no test-based selection"
  ) + theme_w5() + theme(axis.text.x = element_text(angle = 18, hjust = 1))

# W5-R-02: all horizons, overall and burst metrics.
horizon_long <- rbind(
  data.frame(model = metrics$model, horizon_minutes = metrics$horizon_minutes, metric = "MAE", value = metrics$mae),
  data.frame(model = metrics$model, horizon_minutes = metrics$horizon_minutes, metric = "Burst F1", value = metrics$f1)
)
horizon_long$metric <- factor(horizon_long$metric, levels = c("MAE", "Burst F1"))
p2 <- ggplot(horizon_long, aes(x = factor(horizon_minutes, levels = c(5, 15, 60)), y = value, fill = model)) +
  geom_col(position = position_dodge(width = 0.82), width = 0.76, color = "white", linewidth = 0.25) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()), expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "W5-R-02 Overall MAE and burst F1 across horizons",
    subtitle = "The 15-minute task is primary; 5- and 60-minute results are frozen appendix evidence",
    x = "Forecast horizon", y = NULL,
    caption = "Source: table_04*/table_05* frozen result tables"
  ) + theme_w5() + scale_x_discrete(labels = c("5 min", "15 min", "60 min"))
if (max(metrics$f1, na.rm = TRUE) <= 1e-12) {
  p2 <- p2 + geom_text(
    data = data.frame(horizon_minutes = 15, metric = factor("Burst F1", levels = c("MAE", "Burst F1")), value = 0.5, label = "All models and horizons: F1 = 0"),
    aes(x = factor(horizon_minutes, levels = c(5, 15, 60)), y = value, label = label),
    inherit.aes = FALSE, color = MUTED, fontface = "bold", size = 3.2
  )
}

# W5-R-03: representative week.
actual_week <- unique(week[, c("relative_hour", "y_true")])
p3 <- ggplot() +
  geom_line(data = actual_week, aes(x = relative_hour, y = y_true, color = "Actual", linetype = "Actual"), linewidth = 0.75) +
  geom_line(data = week, aes(x = relative_hour, y = y_pred, color = model, linetype = model), linewidth = 0.42, alpha = 0.92) +
  scale_color_manual(values = c("Actual" = INK, model_colors)) +
  scale_linetype_manual(values = c("Actual" = "solid", model_linetypes)) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "W5-R-03 Representative test-week forecasts (h15)",
    subtitle = "Complete synthetic-UTC week objectively selected by median weekly load",
    x = "Hours from representative-week start", y = "Token load (Token/5-min)",
    caption = "Source: test_predictions_all_models.csv | same timestamp + horizon labels for all models"
  ) + theme_w5()

# W5-R-04: burst zoom.
actual_zoom <- unique(zoom[, c("relative_hour", "y_true")])
threshold <- unique(zoom$train_p95_threshold)[[1]]
p4 <- ggplot() +
  geom_hline(yintercept = threshold, color = MUTED, linetype = "dashed", linewidth = 0.45) +
  geom_line(data = actual_zoom, aes(x = relative_hour, y = y_true, color = "Actual", linetype = "Actual"), linewidth = 0.9) +
  geom_line(data = zoom, aes(x = relative_hour, y = y_pred, color = model, linetype = model), linewidth = 0.6) +
  scale_color_manual(values = c("Actual" = INK, model_colors)) +
  scale_linetype_manual(values = c("Actual" = "solid", model_linetypes)) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "W5-R-04 Local burst-window detail (h15)",
    subtitle = "Centered on the maximum observed test-load window; the horizontal line is the train-only P95",
    x = "Hours from burst-window center", y = "Token load (Token/5-min)",
    caption = "Source: test_predictions_all_models.csv and table_01_burst_threshold.csv"
  ) + theme_w5()

# W5-R-05: PR curves.
ap_labels <- setNames(sprintf("%s (AP=%.3f)", main$model, main$pr_auc), as.character(main$model))
p5 <- ggplot(pr_curve, aes(x = recall, y = precision, color = model, linetype = model)) +
  geom_line(linewidth = 0.75) +
  scale_color_manual(values = model_colors, labels = ap_labels) +
  scale_linetype_manual(values = model_linetypes, labels = ap_labels) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "W5-R-05 Burst precision-recall curves (h15)",
    subtitle = "Forecast token load is the ranking score; AP supplements the frozen P95 operating point",
    x = "Recall", y = "Precision",
    caption = "Source: test_predictions_all_models.csv and table_05_burst_results.csv"
  ) + theme_w5()

# W5-R-06: confusion matrices.
confusion <- rbind(
  data.frame(model = main$model, actual = "Non-burst", predicted = "Non-burst", count = main$true_negative),
  data.frame(model = main$model, actual = "Non-burst", predicted = "Burst", count = main$false_positive),
  data.frame(model = main$model, actual = "Burst", predicted = "Non-burst", count = main$false_negative),
  data.frame(model = main$model, actual = "Burst", predicted = "Burst", count = main$true_positive)
)
confusion$model <- factor(confusion$model, levels = model_order)
confusion$actual <- factor(confusion$actual, levels = c("Burst", "Non-burst"))
confusion$predicted <- factor(confusion$predicted, levels = c("Non-burst", "Burst"))
confusion$row_total <- ave(confusion$count, confusion$model, confusion$actual, FUN = sum)
confusion$share <- ifelse(confusion$row_total > 0, confusion$count / confusion$row_total, 0)
confusion$label <- sprintf("%d\n(%.1f%%)", confusion$count, 100 * confusion$share)
p6 <- ggplot(confusion, aes(x = predicted, y = actual, fill = share)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = label), color = INK, size = 3.1) +
  facet_wrap(~model, nrow = 1) +
  scale_fill_gradient(low = "white", high = "#31688E", limits = c(0, 1), name = "Row share") +
  labs(
    title = "W5-R-06 Burst confusion matrices at the frozen P95 (h15)",
    subtitle = "Cells show count and row-normalized share",
    x = "Predicted class", y = "Actual class",
    caption = "Source: table_05_burst_results.csv"
  ) + theme_w5() + theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "right")

# W5-R-07: burst/non-burst MAE and signed bias.
burst_segment <- segment[segment$dimension == "actual_burst", ]
burst_long <- rbind(
  data.frame(model = burst_segment$model, segment = burst_segment$segment, metric = "MAE", value = burst_segment$mae),
  data.frame(model = burst_segment$model, segment = burst_segment$segment, metric = "Mean bias", value = burst_segment$mean_bias_y_pred_minus_y_true)
)
burst_long$metric <- factor(burst_long$metric, levels = c("MAE", "Mean bias"))
p7 <- ggplot(burst_long, aes(x = model, y = value, fill = segment)) +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.35) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70, color = "white", linewidth = 0.25) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Non-burst" = "#9CC2D8", "Burst" = "#C44E52")) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "W5-R-07 Error in burst versus non-burst windows (h15)",
    subtitle = "MAE measures error magnitude; signed bias reveals systematic under- or over-prediction",
    x = NULL, y = "Token per 5-minute target window",
    caption = "Source: table_06_segment_errors.csv"
  ) + theme_w5() + theme(axis.text.x = element_text(angle = 18, hjust = 1))

# W5-R-08: the remaining four pre-specified segment dimensions.
other <- segment[segment$dimension != "actual_burst", ]
dimension_labels <- c(
  "relative_day_phase" = "Relative day phase",
  "relative_week_phase" = "Relative week phase",
  "gpt4_share" = "GPT-4 share",
  "api_share" = "API share"
)
other_long <- rbind(
  data.frame(model = other$model, dimension = other$dimension, segment = other$segment, metric = "MAE", value = other$mae),
  data.frame(model = other$model, dimension = other$dimension, segment = other$segment, metric = "Mean bias", value = other$mean_bias_y_pred_minus_y_true)
)
other_long$dimension_label <- factor(dimension_labels[other_long$dimension], levels = dimension_labels)
other_long$metric <- factor(other_long$metric, levels = c("MAE", "Mean bias"))
p8 <- ggplot(other_long, aes(x = value, y = segment, color = model, shape = model)) +
  geom_vline(xintercept = 0, color = GRID, linewidth = 0.35) +
  geom_point(position = position_dodge(width = 0.55), size = 2.0) +
  facet_grid(rows = vars(dimension_label), cols = vars(metric), scales = "free") +
  scale_color_manual(values = model_colors) + scale_shape_manual(values = c(16, 17, 15, 18)) +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "W5-R-08 Pre-specified segment error profiles (h15)",
    subtitle = "Relative phases are not real weekdays; target-time composition is diagnostic only",
    x = "Token per 5-minute target window", y = NULL,
    caption = "Source: table_06_segment_errors.csv | GPT-4/API target-time shares are never future predictors"
  ) + theme_w5(base_size = 8.8)

# W5-R-09: minimal ablation.
feature_labels <- c(
  "full_features" = "Full features",
  "without_service_structure" = "No service structure",
  "lags_plus_relative_calendar" = "Lags + relative calendar"
)
ablation$feature_label <- factor(feature_labels[ablation$feature_set], levels = feature_labels)
ablation$split <- factor(ablation$split, levels = c("valid", "test"), labels = c("Validation", "Test"))
p9 <- ggplot(ablation, aes(x = feature_label, y = mae, fill = split)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70, color = "white", linewidth = 0.25) +
  geom_text(aes(label = paste0(round(mae / 100) / 10, "k")), position = position_dodge(width = 0.78), vjust = -0.45, size = 3.0, color = INK) +
  scale_fill_manual(values = c("Validation" = "#8FBBD5", "Test" = "#D97706")) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()), expand = expansion(mult = c(0, 0.14))) +
  labs(
    title = "W5-R-09 Minimal XGBoost ablation (h15)",
    subtitle = "Validation MAE alone selects the feature set; test bars are final reporting evidence",
    x = NULL, y = "MAE (Token/5-min)",
    caption = "Source: table_06a_xgb_h15_ablation.csv"
  ) + theme_w5()

# W5-R-10: batch-3 robustness over every horizon.
robust_long <- rbind(
  data.frame(model = robustness$model, horizon_minutes = robustness$horizon_minutes, metric = "MAE", value = robustness$mae),
  data.frame(model = robustness$model, horizon_minutes = robustness$horizon_minutes, metric = "Burst F1", value = robustness$f1)
)
robust_long$metric <- factor(robust_long$metric, levels = c("MAE", "Burst F1"))
p10 <- ggplot(robust_long, aes(x = factor(horizon_minutes, levels = c(5, 15, 60)), y = value, fill = model)) +
  geom_col(position = position_dodge(width = 0.82), width = 0.76, color = "white", linewidth = 0.25) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale()), expand = expansion(mult = c(0, 0.12))) +
  scale_x_discrete(labels = c("5 min", "15 min", "60 min")) +
  labs(
    title = "W5-R-10 BurstGPT-3 external zero-shot robustness",
    subtitle = "Models and train-1/2 P95 transfer without batch-3 fitting or selection",
    x = "Forecast horizon", y = NULL,
    caption = "Source: table_07_robustness_burstgpt3.csv"
  ) + theme_w5()

plots <- list(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
filenames <- c(
  "fig_w5_r_01_h15_regression_accuracy.png",
  "fig_w5_r_02_mae_by_horizon.png",
  "fig_w5_r_03_representative_week_h15.png",
  "fig_w5_r_04_burst_zoom_h15.png",
  "fig_w5_r_05_pr_curves_h15.png",
  "fig_w5_r_06_confusion_matrices_h15.png",
  "fig_w5_r_07_burst_segment_mae_bias_h15.png",
  "fig_w5_r_08_segment_profiles_h15.png",
  "fig_w5_r_09_xgb_ablation_h15.png",
  "fig_w5_r_10_burstgpt3_robustness.png"
)
widths <- c(12.2, 11.8, 13.0, 12.2, 8.8, 13.0, 12.0, 13.2, 10.8, 12.4)
heights <- c(5.4, 5.6, 5.5, 5.6, 6.1, 4.5, 5.3, 12.5, 5.6, 5.4)
paths <- mapply(save_plot, plots, filenames, widths, heights, SIMPLIFY = TRUE)

overview_path <- file.path(figures, "w5_rstudio_overview.png")
png(overview_path, width = 2500, height = 3000, res = 180, bg = "white")
grid.newpage()
pushViewport(viewport(layout = grid.layout(5, 2)))
for (i in seq_along(plots)) {
  row <- ((i - 1) %/% 2) + 1
  column <- ((i - 1) %% 2) + 1
  print(plots[[i]] + theme_w5(base_size = 5.6), vp = viewport(layout.pos.row = row, layout.pos.col = column))
}
dev.off()

manifest <- data.frame(
  figure_number = sprintf("W5-R-%02d", 1:10),
  rstudio_file = filenames,
  exists = file.exists(paths),
  bytes = as.numeric(file.info(paths)$size)
)
write.csv(manifest, file.path(shared, "w5_r_figure_manifest.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(capture.output(sessionInfo()), file.path(shared, "r_session_info.txt"), useBytes = TRUE)
cat("Saved", length(paths), "RStudio Week-5 figures to", figures, "\n")
cat("Saved overview:", overview_path, "\n")
