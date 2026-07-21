# Week-3 visualization bundle for RStudio.
#
# Source this file in RStudio or run it with Rscript from the project root.
# It reads only the reviewed CSVs exported by src/10_visualize_w3_python.py,
# guaranteeing identical values across Python and R figures.

# Codex/PowerShell may inject the POSIX-only locale name C.UTF-8.  Windows R
# cannot translate this project's Chinese path under that locale, so reset to
# the native Windows UTF-8 locale before resolving the script location.
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
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/"))
  }
  candidate <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(candidate)) {
    return(normalizePath(candidate, winslash = "/"))
  }
  return(NULL)
}

script_path <- find_script_path()
root <- if (!is.null(script_path)) dirname(dirname(script_path)) else normalizePath(getwd(), winslash = "/")
if (!file.exists(file.path(root, "outputs", "tables", "w3_visualization"))) {
  stop("Could not locate outputs/tables/w3_visualization; run src/10_visualize_w3_python.py first")
}

viz_tables <- file.path(root, "outputs", "tables", "w3_visualization")
figures <- file.path(root, "outputs", "figures", "w3_rstudio")
dir.create(figures, recursive = TRUE, showWarnings = FALSE)

BLUE <- "#2F6B9A"
BLUE_LIGHT <- "#BFD7EA"
GOLD <- "#D9A441"
ORANGE <- "#D97706"
INK <- "#222222"
MID <- "#6B7280"
GRID <- "#D9DEE5"
OPEN <- "#EFF3F7"

model_colors <- c("Persistence" = MID, "Seasonal Naive" = GOLD, "XGBoost" = BLUE)
split_colors <- c("train" = BLUE, "valid" = GOLD, "test" = ORANGE)

theme_w3 <- function(base_size = 10) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = GRID, linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.line.x = element_line(color = INK, linewidth = 0.35),
      axis.line.y = element_line(color = INK, linewidth = 0.35),
      plot.title = element_text(face = "bold", size = rel(1.35), color = INK, hjust = 0),
      plot.subtitle = element_text(size = rel(0.88), color = MID, margin = margin(b = 8)),
      plot.caption = element_text(size = rel(0.72), color = MID, hjust = 0, margin = margin(t = 8)),
      axis.title = element_text(color = INK),
      axis.text = element_text(color = INK),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      strip.text = element_text(face = "bold", color = INK),
      strip.background = element_rect(fill = OPEN, color = NA)
    )
}

save_plot <- function(plot, filename, width = 11.2, height = 6.0) {
  path <- file.path(figures, filename)
  ggsave(path, plot = plot, width = width, height = height, dpi = 180, bg = "white")
  path
}

split <- read_csv(file.path(viz_tables, "w3_target_time_splits.csv"), show_col_types = FALSE)
categories <- read_csv(file.path(viz_tables, "w3_feature_category_counts.csv"), show_col_types = FALSE)
tuning <- read_csv(file.path(viz_tables, "w3_tuning_candidates.csv"), show_col_types = FALSE)
metrics <- read_csv(file.path(viz_tables, "w3_model_metrics.csv"), show_col_types = FALSE)
diagnostics <- read_csv(file.path(viz_tables, "w3_prediction_diagnostics.csv"), show_col_types = FALSE)
error_analysis <- read_csv(file.path(viz_tables, "w3_h15_error_analysis.csv"), show_col_types = FALSE)
trace <- read_csv(file.path(viz_tables, "w3_h15_representative_day.csv"), show_col_types = FALSE)
native <- read_csv(file.path(viz_tables, "w3_native_importance.csv"), show_col_types = FALSE)
shap_values <- read_csv(file.path(viz_tables, "w3_shap_importance.csv"), show_col_types = FALSE)
top12 <- read_csv(file.path(viz_tables, "w3_importance_top12.csv"), show_col_types = FALSE)

# W3-01: target-time intervals.
split_order <- unlist(lapply(c(5, 15, 60), function(h) paste0("h", h, " ", c("train", "valid", "test"))))
split$row_label <- factor(split$row_label, levels = rev(split_order))
boundary_values <- sort(unique(split$target_start_relative_day[split$split %in% c("valid", "test")]))
p1 <- ggplot(split, aes(y = row_label, color = split)) +
  geom_vline(xintercept = boundary_values, color = INK, linetype = "dotted", linewidth = 0.45) +
  geom_segment(
    aes(x = target_start_relative_day, xend = target_end_relative_day, yend = row_label),
    linewidth = 3.2, lineend = "round"
  ) +
  geom_point(aes(x = target_start_relative_day), size = 2.2) +
  geom_point(aes(x = target_end_relative_day), size = 2.2) +
  geom_text(
    aes(x = target_end_relative_day, label = comma(n_samples)),
    color = INK, hjust = -0.08, size = 3.0, show.legend = FALSE
  ) +
  scale_color_manual(values = split_colors, breaks = c("train", "valid", "test"), labels = c("Train", "Valid", "Test")) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.12))) +
  labs(
    title = "W3-01 Target-time split intervals",
    subtitle = "All nine datasets are assigned by target_time; dotted lines are validation/test boundaries",
    x = "Relative day on the synthetic time axis", y = NULL,
    caption = "Source: table_03_feature_split_summary.csv | all leakage_check values passed"
  ) +
  theme_w3()

# W3-02: feature availability.
categories$category_label <- factor(categories$category_label, levels = categories$category_label[order(categories$n_features)])
p2 <- ggplot(categories, aes(x = n_features, y = category_label)) +
  geom_col(fill = BLUE, color = "#244F70", linewidth = 0.4, width = 0.68) +
  geom_text(aes(label = paste0(n_features, " features")), hjust = -0.08, size = 3.2, color = INK) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title = "W3-02 Feature availability by source",
    subtitle = "36/36 model columns are available at t; future targets and real-calendar labels are excluded",
    x = "Number of model features", y = NULL,
    caption = "Source: table_03_feature_availability.csv"
  ) +
  theme_w3() + theme(legend.position = "none")

# W3-03: tuning trade-off.
tuning$horizon_label <- factor(tuning$horizon_label, levels = c("h5", "h15", "h60"), labels = c("5-minute horizon", "15-minute horizon", "60-minute horizon"))
p3 <- ggplot() +
  geom_point(
    data = tuning[!tuning$selected, ],
    aes(x = valid_mae, y = valid_f1),
    shape = 21, fill = BLUE_LIGHT, color = BLUE, size = 2.5, stroke = 0.6
  ) +
  geom_point(
    data = tuning[tuning$selected, ],
    aes(x = valid_mae, y = valid_f1),
    shape = 21, fill = GOLD, color = INK, size = 3.6, stroke = 0.8
  ) +
  geom_text(
    data = tuning[tuning$selected, ],
    aes(x = valid_mae, y = valid_f1, label = paste0("selected #", candidate_index)),
    hjust = -0.05, vjust = -0.9, size = 2.8, color = INK
  ) +
  facet_wrap(~horizon_label, scales = "free_x", nrow = 1) +
  labs(
    title = "W3-03 Validation tuning trade-off",
    subtitle = "Twelve bounded random candidates per horizon; selection uses minimum validation MAE",
    x = "Validation MAE", y = "Validation F1 at fixed train P95",
    caption = "Source: xgb_tuning_log.csv | test data not used in selection"
  ) +
  theme_w3() + theme(legend.position = "none")

make_mae_plot <- function(split_name, code) {
  formal <- metrics[metrics$split == split_name & metrics$model_label %in% names(model_colors), ]
  formal$model_label <- factor(formal$model_label, levels = names(model_colors))
  formal$horizon_label <- factor(formal$horizon_label, levels = c("h5", "h15", "h60"), labels = c("5 min", "15 min", "60 min"))
  plot <- ggplot(formal, aes(x = horizon_label, y = mae, fill = model_label)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.7, color = "white", linewidth = 0.25) +
    geom_text(
      aes(label = paste0(round(mae / 100) / 10, "k")),
      position = position_dodge(width = 0.78), vjust = -0.45, size = 3.0, color = INK
    ) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(labels = label_number(big.mark = ","), expand = expansion(mult = c(0, 0.14))) +
    labs(
      title = paste0(code, " ", tools::toTitleCase(split_name), " MAE by horizon"),
      subtitle = if (split_name == "valid")
        "Validation-only comparison; selected XGBoost uses the lowest validation MAE"
      else
        "One post-freeze test evaluation; test results were never used for parameter selection",
      x = NULL, y = "MAE (tokens per 5-minute target window)",
      caption = "Source: table_02_baseline_results.csv and table_04_xgb_results.csv"
    ) + theme_w3()
  if (split_name == "valid") {
    default <- metrics[metrics$split == "valid" & metrics$model_label == "XGBoost default", ]
    default$horizon_label <- factor("15 min", levels = c("5 min", "15 min", "60 min"))
    plot <- plot +
      geom_point(data = default, aes(x = horizon_label, y = mae), inherit.aes = FALSE, shape = 23, fill = ORANGE, color = INK, size = 3.4) +
      geom_text(data = default, aes(x = horizon_label, y = mae, label = paste0("default ", round(mae / 100) / 10, "k")), inherit.aes = FALSE, hjust = -0.05, vjust = -0.7, size = 2.8)
  }
  plot
}

p4 <- make_mae_plot("valid", "W3-04")
p5 <- make_mae_plot("test", "W3-05")

# W3-06: representative day trace.
trace_long <- rbind(
  data.frame(relative_hour = trace$relative_hour, model = "Actual", value = trace$actual_token_load),
  data.frame(relative_hour = trace$relative_hour, model = "XGBoost", value = trace$xgboost_prediction),
  data.frame(relative_hour = trace$relative_hour, model = "Persistence", value = trace$persistence_prediction),
  data.frame(relative_hour = trace$relative_hour, model = "Seasonal Naive", value = trace$seasonal_naive_prediction)
)
trace_long$model <- factor(trace_long$model, levels = c("Actual", "XGBoost", "Persistence", "Seasonal Naive"))
trace_colors <- c("Actual" = INK, model_colors)
trace_lines <- c("Actual" = "solid", "XGBoost" = "solid", "Persistence" = "dashed", "Seasonal Naive" = "dotdash")
relative_day <- unique(trace$relative_day)[[1]]
p6 <- ggplot(trace_long, aes(x = relative_hour, y = value, color = model, linetype = model)) +
  geom_hline(yintercept = 0, color = INK, linewidth = 0.35) +
  geom_line(linewidth = 0.55) +
  scale_color_manual(values = trace_colors) +
  scale_linetype_manual(values = trace_lines) +
  scale_x_continuous(limits = c(0, 24), breaks = seq(0, 24, 3)) +
  scale_y_continuous(labels = label_number(big.mark = ",")) +
  labs(
    title = "W3-06 15-minute forecasts on a representative test day",
    subtitle = paste0("Complete relative day ", relative_day, ", selected by median daily load; raw negative XGBoost values are retained"),
    x = "Hour within the selected relative day", y = "Tokens per 5-minute target window",
    caption = "Source: pred_xgb_test_h15.csv and pred_baselines_test.csv"
  ) + theme_w3()

# W3-07: load shift and mean bias.
error_order <- c("Valid | XGBoost", "Test | XGBoost", "Test | Persistence", "Test | Seasonal Naive")
error_analysis$display_label <- factor(error_analysis$display_label, levels = rev(error_order))
p7 <- ggplot(error_analysis, aes(y = display_label)) +
  geom_segment(aes(x = actual_mean / 1000, xend = prediction_mean / 1000, yend = display_label), color = GRID, linewidth = 2.2) +
  geom_point(aes(x = actual_mean / 1000), color = INK, size = 2.8) +
  geom_point(aes(x = prediction_mean / 1000), shape = 15, color = BLUE, size = 3.0) +
  geom_text(
    aes(x = pmax(actual_mean, prediction_mean) / 1000, label = paste0("bias ", sprintf("%+.1fk", mean_error_bias / 1000))),
    hjust = -0.08, size = 3.0, color = INK
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.23))) +
  labs(
    title = "W3-07 Validation-to-test load shift and mean bias",
    subtitle = "Test actual mean is 74.86% below validation; Persistence nearly matches the test mean",
    x = "Mean token load / prediction (thousands per 5-minute window)", y = NULL,
    caption = "Source: table_04_xgb_h15_error_analysis.csv | test rows are interpretation only"
  ) + theme_w3() + theme(legend.position = "none")

# W3-08: negative predictions.
diagnostics$stage_order <- match(diagnostics$stage_label, c("Default valid", "Selected valid", "Final test"))
diagnostics <- diagnostics[order(diagnostics$horizon_minutes, diagnostics$stage_order), ]
diagnostics$display_label <- factor(diagnostics$display_label, levels = rev(diagnostics$display_label))
p8 <- ggplot(diagnostics, aes(x = 100 * negative_prediction_rate, y = display_label, fill = stage_label)) +
  geom_col(width = 0.68, color = INK, linewidth = 0.25) +
  geom_text(
    aes(label = paste0(comma(negative_prediction_count), " (", sprintf("%.1f", 100 * negative_prediction_rate), "%)")),
    hjust = -0.05, size = 3.0, color = INK
  ) +
  scale_fill_manual(values = c("Default valid" = ORANGE, "Selected valid" = BLUE, "Final test" = BLUE)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title = "W3-08 Negative-prediction diagnostic",
    subtitle = "Every series is non-constant and target-time aligned; raw values are shown without post-test clipping",
    x = "Share of raw predictions below zero (%)", y = NULL,
    caption = "Source: table_04_xgb_prediction_diagnostics.csv"
  ) + theme_w3() + theme(legend.position = "none")

# W3-09: native and SHAP importance heatmaps.
native_panel <- native[native$top12_overall, c("feature", "horizon_minutes", "normalized_importance")]
native_panel$importance_type <- "Native feature_importances_"
shap_panel <- shap_values[shap_values$top12_overall, c("feature", "horizon_minutes", "normalized_importance")]
shap_panel$importance_type <- "Mean absolute SHAP"
importance <- rbind(native_panel, shap_panel)
importance$feature <- factor(importance$feature, levels = rev(top12$feature))
importance$horizon <- factor(paste0("h", importance$horizon_minutes), levels = c("h5", "h15", "h60"))
importance$importance_type <- factor(importance$importance_type, levels = c("Native feature_importances_", "Mean absolute SHAP"))
p9 <- ggplot(importance, aes(x = horizon, y = feature, fill = normalized_importance)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", normalized_importance)), size = 2.7, color = INK) +
  facet_wrap(~importance_type, nrow = 1) +
  scale_fill_gradient(low = "#F7FAFC", high = BLUE, limits = c(0, 1), name = "Normalized\nimportance") +
  labs(
    title = "W3-09 Feature-importance evidence across horizons",
    subtitle = "Top 12 features by average rank across native and SHAP evidence; all passed availability audit",
    x = NULL, y = NULL,
    caption = "Source: xgb_feature_importance_h*.csv and xgb_shap_importance_h*.csv"
  ) + theme_w3() + theme(legend.position = "right", panel.grid = element_blank())

plots <- list(p1, p2, p3, p4, p5, p6, p7, p8, p9)
filenames <- c(
  "fig_w3_r_01_target_time_splits.png",
  "fig_w3_r_02_feature_availability.png",
  "fig_w3_r_03_tuning_tradeoff.png",
  "fig_w3_r_04_validation_mae.png",
  "fig_w3_r_05_test_mae.png",
  "fig_w3_r_06_h15_representative_day.png",
  "fig_w3_r_07_distribution_shift.png",
  "fig_w3_r_08_prediction_diagnostics.png",
  "fig_w3_r_09_importance_matrix.png"
)
widths <- c(11.5, 10.2, 13.6, 11.2, 11.2, 12.2, 11.2, 11.2, 12.7)
heights <- c(6.4, 5.7, 4.9, 6.0, 6.0, 6.0, 5.8, 6.0, 7.0)
paths <- mapply(save_plot, plots, filenames, widths, heights, SIMPLIFY = TRUE)

# A 3x3 native grid overview avoids requiring gridExtra or the png package.
overview_path <- file.path(figures, "w3_rstudio_overview.png")
png(overview_path, width = 2400, height = 2400, res = 180, bg = "white")
grid.newpage()
pushViewport(viewport(layout = grid.layout(3, 3)))
for (i in seq_along(plots)) {
  row <- ((i - 1) %/% 3) + 1
  column <- ((i - 1) %% 3) + 1
  print(plots[[i]] + theme_w3(base_size = 5.8), vp = viewport(layout.pos.row = row, layout.pos.col = column))
}
dev.off()

manifest <- data.frame(
  figure_number = sprintf("W3-%02d", 1:9),
  rstudio_file = filenames,
  exists = file.exists(paths),
  bytes = as.numeric(file.info(paths)$size)
)
write.csv(manifest, file.path(viz_tables, "w3_r_figure_manifest.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(capture.output(sessionInfo()), file.path(viz_tables, "r_session_info.txt"), useBytes = TRUE)

cat("Saved", length(paths), "RStudio W3 figures to", figures, "\n")
cat("Saved overview:", overview_path, "\n")
