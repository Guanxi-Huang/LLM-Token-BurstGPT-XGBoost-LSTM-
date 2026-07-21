# Week-4 LSTM visualization bundle for RStudio.
# Run src/12_visualize_w4_python.py first; both toolchains use the same CSVs.

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
viz_tables <- file.path(root, "outputs", "tables", "w4_visualization")
if (!dir.exists(viz_tables)) stop("Run src/12_visualize_w4_python.py before this RStudio script")
figures <- file.path(root, "outputs", "figures", "w4_rstudio")
dir.create(figures, recursive = TRUE, showWarnings = FALSE)

BLUE <- "#2F6B9A"; BLUE_LIGHT <- "#BFD7EA"; GOLD <- "#D9A441"
ORANGE <- "#D97706"; INK <- "#222222"; MID <- "#6B7280"
GRID <- "#D9DEE5"; OPEN <- "#EFF3F7"
model_order <- c("Persistence", "Seasonal Naive", "XGBoost", "LSTM")
model_colors <- c("Persistence" = MID, "Seasonal Naive" = GOLD, "XGBoost" = ORANGE, "LSTM" = BLUE)
model_lines <- c("Persistence" = "dashed", "Seasonal Naive" = "dotdash", "XGBoost" = "dotted", "LSTM" = "solid")
split_colors <- c("train" = BLUE, "valid" = GOLD, "test" = ORANGE)

theme_w4 <- function(base_size = 10) {
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
      axis.title = element_text(color = INK), axis.text = element_text(color = INK),
      legend.position = "top", legend.justification = "left", legend.title = element_blank(),
      strip.text = element_text(face = "bold", color = INK),
      strip.background = element_rect(fill = OPEN, color = NA)
    )
}

save_plot <- function(plot, filename, width = 11.2, height = 6.0) {
  path <- file.path(figures, filename)
  ggsave(path, plot = plot, width = width, height = height, dpi = 180, bg = "white")
  path
}

audit <- read_csv(file.path(viz_tables, "w4_sequence_audit.csv"), show_col_types = FALSE)
feature_counts <- read_csv(file.path(viz_tables, "w4_feature_counts.csv"), show_col_types = FALSE)
history <- read_csv(file.path(viz_tables, "w4_training_history.csv"), show_col_types = FALSE)
selected <- read_csv(file.path(viz_tables, "w4_selected_lstm.csv"), show_col_types = FALSE)
metrics <- read_csv(file.path(viz_tables, "w4_model_metrics.csv"), show_col_types = FALSE)
trace <- read_csv(file.path(viz_tables, "w4_h15_representative_day.csv"), show_col_types = FALSE)
bias <- read_csv(file.path(viz_tables, "w4_h15_test_bias.csv"), show_col_types = FALSE)

# W4-01: sequence audit.
audit$display_label <- paste0("h", audit$horizon_minutes, " | ", audit$split)
sequence_order <- unlist(lapply(c(5, 15, 60), function(h) paste0("h", h, " | ", c("train", "valid", "test"))))
audit$display_label <- factor(audit$display_label, levels = rev(sequence_order))
p1 <- ggplot(audit, aes(x = n_samples, y = display_label, fill = split)) +
  geom_col(width = 0.68, color = "white") +
  geom_text(aes(label = comma(n_samples)), hjust = -0.08, size = 3.0, color = INK) +
  scale_fill_manual(values = split_colors) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.17))) +
  labs(title = "W4-01 Audited LSTM sequence counts",
       subtitle = "144 prior 5-minute windows; latest input ends 5 minutes before forecast origin",
       x = "Sequence samples", y = NULL,
       caption = "Source: table_05_lstm_sequence_audit.csv | all labels confined to their assigned split") +
  theme_w4()

# W4-02: input composition.
feature_counts$category <- factor(feature_counts$category, levels = feature_counts$category[order(feature_counts$n_features)])
p2 <- ggplot(feature_counts, aes(x = n_features, y = category, fill = category)) +
  geom_col(width = 0.68, color = INK, linewidth = 0.3) +
  geom_text(aes(label = paste0(n_features, " features")), hjust = -0.08, size = 3.2, color = INK) +
  scale_fill_manual(values = c("Historical load" = BLUE, "Request structure" = GOLD, "Relative phase" = ORANGE)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.28))) +
  labs(title = "W4-02 LSTM input composition",
       subtitle = "Ten prediction-time features are repeated across a 144-step (12-hour) lookback",
       x = "Number of model inputs per 5-minute step", y = NULL,
       caption = "Source: frozen configs/lstm_h5.yaml, lstm_h15.yaml, and lstm_h60.yaml") +
  theme_w4() + theme(legend.position = "none")

# W4-03: validation histories.
history$horizon_label <- factor(paste0("h", history$horizon_minutes), levels = c("h5", "h15", "h60"), labels = c("5-minute horizon", "15-minute horizon", "60-minute horizon"))
history_long <- rbind(
  data.frame(epoch = history$epoch, horizon_label = history$horizon_label, series = "Train", value = history$train_mae_scaled),
  data.frame(epoch = history$epoch, horizon_label = history$horizon_label, series = "Validation", value = history$valid_mae_scaled)
)
best_lines <- data.frame(
  horizon_label = factor(paste0("h", selected$horizon_minutes), levels = c("h5", "h15", "h60"), labels = c("5-minute horizon", "15-minute horizon", "60-minute horizon")),
  best_epoch = selected$best_epoch
)
p3 <- ggplot(history_long, aes(x = epoch, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 0.65) +
  geom_vline(data = best_lines, aes(xintercept = best_epoch), inherit.aes = FALSE, color = INK, linetype = "dotted", linewidth = 0.45) +
  facet_wrap(~horizon_label, scales = "free", nrow = 1) +
  scale_color_manual(values = c("Train" = BLUE, "Validation" = GOLD)) +
  scale_linetype_manual(values = c("Train" = "solid", "Validation" = "dashed")) +
  labs(title = "W4-03 Validation training histories",
       subtitle = "Dotted line marks the validation-selected best epoch before the frozen final refit",
       x = "Epoch", y = "Scaled MAE",
       caption = "Source: lstm_history_h5.csv, lstm_history_h15.csv, and lstm_history_h60.csv") +
  theme_w4()

make_mae_plot <- function(split_name, code) {
  frame <- metrics[metrics$split == split_name & metrics$model_label %in% model_order, ]
  frame$model_label <- factor(frame$model_label, levels = model_order)
  frame$horizon_label <- factor(paste0("h", frame$horizon_minutes), levels = c("h5", "h15", "h60"), labels = c("5 min", "15 min", "60 min"))
  ggplot(frame, aes(x = horizon_label, y = mae, fill = model_label)) +
    geom_col(position = position_dodge(width = 0.82), width = 0.76, color = "white", linewidth = 0.25) +
    geom_text(aes(label = paste0(sprintf("%.1f", mae / 1000), "k")), position = position_dodge(width = 0.82), vjust = -0.45, size = 2.8, color = INK) +
    scale_fill_manual(values = model_colors) +
    scale_y_continuous(labels = label_number(big.mark = ","), expand = expansion(mult = c(0, 0.15))) +
    labs(title = paste0(code, " ", tools::toTitleCase(split_name), " MAE by horizon"),
         subtitle = if (split_name == "valid") "Validation-only model selection evidence" else "Single post-freeze test evaluation; no test-informed retuning",
         x = NULL, y = "MAE (tokens per 5-minute target window)",
         caption = "Source: baseline, XGBoost, and W4 LSTM metric tables") +
    theme_w4()
}
p4 <- make_mae_plot("valid", "W4-04")
p5 <- make_mae_plot("test", "W4-05")

# W4-06: representative h15 day.
trace_long <- rbind(
  data.frame(relative_hour = trace$relative_hour, model = "Actual", value = trace$actual_token_load),
  data.frame(relative_hour = trace$relative_hour, model = "Persistence", value = trace$persistence_prediction),
  data.frame(relative_hour = trace$relative_hour, model = "Seasonal Naive", value = trace$seasonal_naive_prediction),
  data.frame(relative_hour = trace$relative_hour, model = "XGBoost", value = trace$xgboost_prediction),
  data.frame(relative_hour = trace$relative_hour, model = "LSTM", value = trace$lstm_prediction)
)
trace_long$model <- factor(trace_long$model, levels = c("Actual", model_order))
trace_colors <- c("Actual" = INK, model_colors)
trace_lines <- c("Actual" = "solid", model_lines)
relative_day <- unique(trace$relative_day)[[1]]
p6 <- ggplot(trace_long, aes(x = relative_hour, y = value, color = model, linetype = model)) +
  geom_line(linewidth = 0.55) +
  scale_color_manual(values = trace_colors) + scale_linetype_manual(values = trace_lines) +
  scale_x_continuous(limits = c(0, 24), breaks = seq(0, 24, 3)) +
  scale_y_continuous(labels = label_number(big.mark = ",")) +
  labs(title = "W4-06 15-minute forecasts on a representative test day",
       subtitle = paste0("Complete relative day ", relative_day, ", selected by median daily load among complete test days"),
       x = "Hour within the selected relative day", y = "Tokens per 5-minute target window",
       caption = "Source: aligned LSTM, XGBoost, and baseline h15 predictions") + theme_w4()

# W4-07: h15 mean bias.
bias$model_label <- factor(bias$model_label, levels = rev(model_order))
p7 <- ggplot(bias, aes(y = model_label)) +
  geom_segment(aes(x = actual_mean / 1000, xend = prediction_mean / 1000, yend = model_label), color = GRID, linewidth = 2.2) +
  geom_point(aes(x = actual_mean / 1000), color = INK, size = 2.8) +
  geom_point(aes(x = prediction_mean / 1000, color = model_label), shape = 15, size = 3.0) +
  geom_text(aes(x = pmax(actual_mean, prediction_mean) / 1000, label = paste0("bias ", sprintf("%+.1fk", mean_error_bias / 1000))), hjust = -0.08, size = 3.0, color = INK) +
  scale_color_manual(values = model_colors) + scale_x_continuous(expand = expansion(mult = c(0.02, 0.24))) +
  labs(title = "W4-07 15-minute test mean bias",
       subtitle = "Same 5,228 target windows for all methods; direction matters in the low-load test regime",
       x = "Mean tokens per 5-minute target window (thousands)", y = NULL,
       caption = "Source: aligned h15 prediction CSVs; test analysis is descriptive only") +
  theme_w4() + theme(legend.position = "none")

# W4-08: fixed-P95 burst counts.
burst <- metrics[metrics$split == "test" & metrics$model_label %in% model_order, ]
burst$model_label <- factor(burst$model_label, levels = model_order)
burst$horizon_label <- factor(paste0("h", burst$horizon_minutes), levels = c("h5", "h15", "h60"), labels = c("5 min", "15 min", "60 min"))
actual_count <- unique(burst$actual_burst_windows)[[1]]
all_zero_f1 <- all(abs(burst$f1) < 1e-12)
p8 <- ggplot(burst, aes(x = horizon_label, y = predicted_burst_windows, fill = model_label)) +
  geom_hline(yintercept = actual_count, color = INK, linetype = "dashed", linewidth = 0.5) +
  geom_col(position = position_dodge(width = 0.82), width = 0.76, color = "white", linewidth = 0.25) +
  geom_text(aes(label = predicted_burst_windows), position = position_dodge(width = 0.82), vjust = -0.4, size = 2.9, color = INK) +
  scale_fill_manual(values = model_colors) + scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "W4-08 Fixed-P95 burst outcomes",
       subtitle = if (all_zero_f1) paste0("Dashed line is actual count (", actual_count, "); predicted counts do not imply hits and every test F1 is zero") else paste0("Dashed line is actual count (", actual_count, "); consult the metric table for precision, recall, and F1"),
       x = NULL, y = "Predicted burst windows",
       caption = "Source: one-time test metrics | threshold fixed from training token-load P95") + theme_w4()

# W4-09: frozen training profile.
profile <- rbind(
  data.frame(horizon_minutes = selected$horizon_minutes, metric = "Train MAE", value = selected$train_mae_tokens),
  data.frame(horizon_minutes = selected$horizon_minutes, metric = "Validation MAE", value = selected$valid_mae_tokens)
)
profile$horizon_label <- factor(paste0("h", profile$horizon_minutes), levels = rev(c("h5", "h15", "h60")))
profile$metric <- factor(profile$metric, levels = c("Train MAE", "Validation MAE"))
best_labels <- selected
best_labels$horizon_label <- factor(paste0("h", best_labels$horizon_minutes), levels = rev(c("h5", "h15", "h60")))
p9 <- ggplot(profile, aes(x = value / 1000, y = horizon_label, group = horizon_label)) +
  geom_line(color = GRID, linewidth = 2.4) +
  geom_point(aes(color = metric, shape = metric), size = 3.2) +
  geom_text(data = best_labels, aes(x = pmax(train_mae_tokens, valid_mae_tokens) / 1000, y = horizon_label, label = paste0("best epoch ", best_epoch)), inherit.aes = FALSE, hjust = -0.08, size = 3.0, color = INK) +
  scale_color_manual(values = c("Train MAE" = BLUE, "Validation MAE" = GOLD)) +
  scale_shape_manual(values = c("Train MAE" = 16, "Validation MAE" = 15)) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.25))) +
  labs(title = "W4-09 Frozen LSTM training profile",
       subtitle = "One 5,537-parameter LSTM(32) candidate per horizon; final refit uses the selected best epoch",
       x = "MAE (thousands of tokens per target window)", y = NULL,
       caption = "Source: lstm_tuning_log.csv | selection_data = validation only") + theme_w4()

plots <- list(p1, p2, p3, p4, p5, p6, p7, p8, p9)
filenames <- c(
  "fig_w4_r_01_sequence_audit.png", "fig_w4_r_02_feature_composition.png",
  "fig_w4_r_03_training_history.png", "fig_w4_r_04_validation_mae.png",
  "fig_w4_r_05_test_mae.png", "fig_w4_r_06_h15_representative_day.png",
  "fig_w4_r_07_h15_test_bias.png", "fig_w4_r_08_burst_outcomes.png",
  "fig_w4_r_09_training_profile.png"
)
widths <- c(11.2, 10.2, 13.7, 11.4, 11.4, 12.3, 10.8, 11.4, 10.8)
heights <- c(6.2, 5.5, 4.9, 6.1, 6.1, 6.1, 5.6, 6.0, 5.5)
paths <- mapply(save_plot, plots, filenames, widths, heights, SIMPLIFY = TRUE)

overview_path <- file.path(figures, "w4_rstudio_overview.png")
png(overview_path, width = 2400, height = 2400, res = 180, bg = "white")
grid.newpage(); pushViewport(viewport(layout = grid.layout(3, 3)))
for (i in seq_along(plots)) {
  row <- ((i - 1) %/% 3) + 1; column <- ((i - 1) %% 3) + 1
  print(plots[[i]] + theme_w4(base_size = 5.8), vp = viewport(layout.pos.row = row, layout.pos.col = column))
}
dev.off()

manifest <- data.frame(figure_number = sprintf("W4-%02d", 1:9), rstudio_file = filenames,
                       exists = file.exists(paths), bytes = as.numeric(file.info(paths)$size))
write.csv(manifest, file.path(viz_tables, "w4_r_figure_manifest.csv"), row.names = FALSE, fileEncoding = "UTF-8")
writeLines(capture.output(sessionInfo()), file.path(viz_tables, "r_session_info.txt"), useBytes = TRUE)
cat("Saved", length(paths), "RStudio W4 figures to", figures, "\n")
cat("Saved overview:", overview_path, "\n")
