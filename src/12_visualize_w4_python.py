"""Create the complete Week-4 LSTM visualization bundle with Python.

The script exports reviewed plotting CSVs for the companion RStudio script.
Both toolchains therefore use the same temporal audit, training histories,
validation metrics, one-time test metrics, and prediction traces.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import yaml


TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures" / "w4_python"
VIZ_TABLES = TABLES / "w4_visualization"
DATA = ROOT / "data" / "processed"

BLUE = "#2F6B9A"
BLUE_LIGHT = "#BFD7EA"
GOLD = "#D9A441"
ORANGE = "#D97706"
INK = "#222222"
MID = "#6B7280"
GRID = "#D9DEE5"
OPEN = "#EFF3F7"

MODEL_ORDER = ["Persistence", "Seasonal Naive", "XGBoost", "LSTM"]
MODEL_COLORS = {
    "Persistence": MID,
    "Seasonal Naive": GOLD,
    "XGBoost": ORANGE,
    "LSTM": BLUE,
}
MODEL_LINES = {
    "Persistence": "--",
    "Seasonal Naive": "-.",
    "XGBoost": ":",
    "LSTM": "-",
}
SPLIT_COLORS = {"train": BLUE, "valid": GOLD, "test": ORANGE}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def title(ax: plt.Axes, heading: str, subtitle: str) -> None:
    ax.set_title(heading, loc="left", fontsize=14, fontweight="bold", pad=30)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=MID)


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=7.5, color=MID)


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def model_label(value: str) -> str:
    mapping = {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal Naive",
        "xgboost_selected": "XGBoost",
        "xgboost_final": "XGBoost",
        "lstm_final": "LSTM",
    }
    if value not in mapping:
        raise ValueError(f"Unexpected model label: {value}")
    return mapping[value]


def export_shared_data() -> dict[str, object]:
    VIZ_TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(TABLES / "table_05_lstm_sequence_audit.csv")
    audit["horizon_label"] = audit["horizon_minutes"].map(lambda v: f"h{int(v)}")
    audit["split_label"] = audit["split"].str.capitalize()
    audit.to_csv(VIZ_TABLES / "w4_sequence_audit.csv", index=False)

    manual = pd.read_csv(TABLES / "table_05_lstm_manual_sequence_check.csv")
    manual.to_csv(VIZ_TABLES / "w4_manual_sequence_check.csv", index=False)

    feature_rows: list[dict[str, object]] = []
    expected_features: list[str] | None = None
    for horizon in (5, 15, 60):
        with (ROOT / "configs" / f"lstm_h{horizon}.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        features = list(config["features"])
        if expected_features is None:
            expected_features = features
        elif features != expected_features:
            raise AssertionError("LSTM feature list differs across horizons")
        category_map = {
            "token_load": "Historical load",
            "request_count": "Request structure",
            "mean_request_tokens": "Request structure",
            "mean_response_tokens": "Request structure",
            "gpt4_share": "Request structure",
            "api_share": "Request structure",
            "relative_day_sin": "Relative phase",
            "relative_day_cos": "Relative phase",
            "relative_week_sin": "Relative phase",
            "relative_week_cos": "Relative phase",
        }
        for feature in features:
            if feature not in category_map:
                raise AssertionError(f"Unclassified LSTM feature: {feature}")
            feature_rows.append(
                {"horizon_minutes": horizon, "feature": feature, "category": category_map[feature]}
            )
    features = pd.DataFrame(feature_rows)
    feature_counts = (
        features.drop_duplicates(["feature", "category"])
        .groupby("category", as_index=False)
        .agg(n_features=("feature", "size"))
        .sort_values("n_features", ascending=False)
    )
    features.to_csv(VIZ_TABLES / "w4_features.csv", index=False)
    feature_counts.to_csv(VIZ_TABLES / "w4_feature_counts.csv", index=False)

    histories: list[pd.DataFrame] = []
    for horizon in (5, 15, 60):
        history = pd.read_csv(TABLES / f"lstm_history_h{horizon}.csv")
        history["horizon_minutes"] = horizon
        histories.append(history)
    history_all = pd.concat(histories, ignore_index=True)
    history_all.to_csv(VIZ_TABLES / "w4_training_history.csv", index=False)

    tuning = pd.read_csv(TABLES / "lstm_tuning_log.csv")
    selected = tuning.loc[tuning["selected"].astype(bool)].copy()
    if len(selected) != 3:
        raise AssertionError("Expected one selected LSTM candidate per horizon")
    selected["horizon_label"] = selected["horizon_minutes"].map(lambda v: f"h{int(v)}")
    selected.to_csv(VIZ_TABLES / "w4_selected_lstm.csv", index=False)

    baseline = pd.read_csv(TABLES / "table_02_baseline_results.csv")
    xgb = pd.read_csv(TABLES / "table_04_xgb_results.csv")
    lstm = pd.read_csv(TABLES / "table_05_lstm_results.csv")
    metric_columns = [
        "split", "model", "horizon_minutes", "n_observations", "mae", "rmse",
        "smape_percent", "precision", "recall", "f1", "actual_burst_windows",
        "predicted_burst_windows", "train_p95_threshold",
    ]
    if "train_p95_threshold" not in lstm.columns and "burst_threshold" in lstm.columns:
        lstm = lstm.rename(columns={"burst_threshold": "train_p95_threshold"})
    def metric_view(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
        subset = frame.loc[frame["model"].isin(models)].copy()
        for column in metric_columns:
            if column not in subset.columns:
                subset[column] = np.nan
        return subset[metric_columns]
    formal = pd.concat(
        [
            metric_view(baseline, ["persistence", "seasonal_naive"]),
            metric_view(xgb, ["xgboost_selected", "xgboost_final"]),
            metric_view(lstm, ["lstm_final"]),
        ],
        ignore_index=True,
    )
    formal["model_label"] = formal["model"].map(model_label)
    formal["horizon_label"] = formal["horizon_minutes"].map(lambda v: f"h{int(v)}")

    lstm_valid = selected.rename(
        columns={
            "valid_mae_tokens": "mae",
            "valid_rmse_tokens": "rmse",
            "valid_f1": "f1",
        }
    ).copy()
    lstm_valid["split"] = "valid"
    lstm_valid["model"] = "lstm_final"
    lstm_valid["model_label"] = "LSTM"
    lstm_valid["horizon_label"] = lstm_valid["horizon_minutes"].map(lambda v: f"h{int(v)}")
    for column in formal.columns:
        if column not in lstm_valid.columns:
            lstm_valid[column] = np.nan
    formal = pd.concat([formal, lstm_valid[formal.columns]], ignore_index=True)
    formal = formal.loc[
        formal["model_label"].isin(MODEL_ORDER)
        & formal["split"].isin(["valid", "test"])
    ].copy()
    formal.to_csv(VIZ_TABLES / "w4_model_metrics.csv", index=False)

    lstm_pred = pd.read_csv(
        TABLES / "pred_lstm_test_h15.csv",
        parse_dates=["feature_time", "input_end_time", "target_time"],
    )
    xgb_pred = pd.read_csv(TABLES / "pred_xgb_test_h15.csv", parse_dates=["target_time"])
    base_pred = pd.read_csv(TABLES / "pred_baselines_test.csv", parse_dates=["timestamp"])
    base_pred = base_pred.rename(columns={"timestamp": "target_time"})
    trace = lstm_pred[
        ["target_time", "actual_token_load", "predicted_token_load"]
    ].rename(columns={"predicted_token_load": "lstm_prediction"})
    trace = trace.merge(
        xgb_pred[["target_time", "actual_token_load", "predicted_token_load"]].rename(
            columns={"actual_token_load": "actual_xgb", "predicted_token_load": "xgboost_prediction"}
        ),
        on="target_time", how="inner", validate="one_to_one",
    )
    trace = trace.merge(
        base_pred[["target_time", "actual_token_load", "pred_persistence_h15", "pred_seasonal_h15"]].rename(
            columns={"actual_token_load": "actual_baseline", "pred_persistence_h15": "persistence_prediction", "pred_seasonal_h15": "seasonal_naive_prediction"}
        ),
        on="target_time", how="inner", validate="one_to_one",
    )
    np.testing.assert_allclose(trace["actual_token_load"], trace["actual_xgb"], rtol=0, atol=0)
    np.testing.assert_allclose(trace["actual_token_load"], trace["actual_baseline"], rtol=0, atol=0)
    series = pd.read_parquet(DATA / "series_5min.parquet", columns=["token_load"])
    series.index = pd.to_datetime(series.index, utc=True)
    origin = series.index.min()
    trace["relative_day"] = ((trace["target_time"] - origin) // pd.Timedelta(days=1)).astype(int)
    daily = trace.groupby("relative_day").agg(n_rows=("actual_token_load", "size"), total_load=("actual_token_load", "sum"))
    complete = daily.loc[daily["n_rows"].eq(288)]
    if complete.empty:
        raise AssertionError("No complete h15 test day")
    representative_day = int((complete["total_load"] - complete["total_load"].median()).abs().idxmin())
    representative = trace.loc[trace["relative_day"].eq(representative_day)].copy()
    day_start = origin + pd.Timedelta(days=representative_day)
    representative["relative_hour"] = (representative["target_time"] - day_start) / pd.Timedelta(hours=1)
    representative[
        ["target_time", "relative_day", "relative_hour", "actual_token_load", "persistence_prediction", "seasonal_naive_prediction", "xgboost_prediction", "lstm_prediction"]
    ].to_csv(VIZ_TABLES / "w4_h15_representative_day.csv", index=False)

    bias_rows: list[dict[str, object]] = []
    for label, column in {
        "Persistence": "persistence_prediction",
        "Seasonal Naive": "seasonal_naive_prediction",
        "XGBoost": "xgboost_prediction",
        "LSTM": "lstm_prediction",
    }.items():
        bias_rows.append(
            {
                "model_label": label,
                "actual_mean": trace["actual_token_load"].mean(),
                "prediction_mean": trace[column].mean(),
                "mean_error_bias": (trace[column] - trace["actual_token_load"]).mean(),
            }
        )
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(VIZ_TABLES / "w4_h15_test_bias.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "representative_relative_day": representative_day,
        "source_policy": "Validated W4 artifacts and frozen W2/W3 comparators only",
    }
    (VIZ_TABLES / "python_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return {
        "audit": audit,
        "feature_counts": feature_counts,
        "history": history_all,
        "selected": selected,
        "metrics": formal,
        "representative": representative,
        "representative_day": representative_day,
        "bias": bias,
    }


def plot_sequence_audit(data: dict[str, object]) -> Path:
    frame = data["audit"].copy()  # type: ignore[union-attr]
    rows = [(h, s) for h in (5, 15, 60) for s in ("train", "valid", "test")]
    values = [int(frame.loc[frame["horizon_minutes"].eq(h) & frame["split"].eq(s), "n_samples"].iloc[0]) for h, s in rows]
    labels = [f"h{h} | {s}" for h, s in rows]
    colors = [SPLIT_COLORS[s] for _, s in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    bars = ax.barh(y, values, color=colors, edgecolor="white")
    ax.set_yticks(y, labels)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontsize=8)
    ax.set_xlabel("Sequence samples")
    ax.set_xlim(0, max(values) * 1.16)
    title(ax, "W4-01 Audited LSTM sequence counts", "Each sample uses 144 prior 5-minute windows; the latest input ends 5 minutes before forecast origin")
    footer(fig, "Source: table_05_lstm_sequence_audit.csv | all labels confined to their assigned split")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_01_sequence_audit.png")


def plot_feature_composition(data: dict[str, object]) -> Path:
    frame = data["feature_counts"].sort_values("n_features")  # type: ignore[union-attr]
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    bars = ax.barh(frame["category"], frame["n_features"], color=[BLUE, GOLD, ORANGE], edgecolor=INK, linewidth=0.4)
    ax.bar_label(bars, labels=[f"{int(v)} features" for v in frame["n_features"]], padding=5)
    ax.set_xlabel("Number of model inputs per 5-minute step")
    ax.set_xlim(0, frame["n_features"].max() * 1.35)
    title(ax, "W4-02 LSTM input composition", "Ten prediction-time features are repeated across a 144-step (12-hour) lookback")
    footer(fig, "Source: configs/lstm_h5.yaml, lstm_h15.yaml, and lstm_h60.yaml")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_02_feature_composition.png")


def plot_training_history(data: dict[str, object]) -> Path:
    history = data["history"]  # type: ignore[assignment]
    selected = data["selected"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.9), sharey=False)
    for ax, horizon in zip(axes, (5, 15, 60)):
        subset = history.loc[history["horizon_minutes"].eq(horizon)]
        row = selected.loc[selected["horizon_minutes"].eq(horizon)].iloc[0]
        ax.plot(subset["epoch"], subset["train_mae_scaled"], color=BLUE, label="Train")
        ax.plot(subset["epoch"], subset["valid_mae_scaled"], color=GOLD, linestyle="--", label="Validation")
        ax.axvline(int(row["best_epoch"]), color=INK, linestyle=":", linewidth=1)
        ax.set_title(f"{horizon}-minute horizon", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Scaled MAE")
    axes[0].legend(loc="upper right")
    fig.suptitle("W4-03 Validation training histories", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.91, "Dotted line marks the validation-selected best epoch before the frozen final refit", color=MID, fontsize=8.8)
    footer(fig, "Source: lstm_history_h5.csv, lstm_history_h15.csv, and lstm_history_h60.csv")
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    return save(fig, "fig_w4_py_03_training_history.png")


def plot_mae(data: dict[str, object], split_name: str, code: str, filename: str) -> Path:
    frame = data["metrics"]  # type: ignore[assignment]
    frame = frame.loc[frame["split"].eq(split_name) & frame["model_label"].isin(MODEL_ORDER)].copy()
    x = np.arange(3)
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.4, 6.1))
    for i, model in enumerate(MODEL_ORDER):
        values = [float(frame.loc[frame["horizon_minutes"].eq(h) & frame["model_label"].eq(model), "mae"].iloc[0]) for h in (5, 15, 60)]
        bars = ax.bar(x + (i - 1.5) * width, values, width, color=MODEL_COLORS[model], edgecolor=INK if model == "LSTM" else "white", linewidth=0.7, label=model)
        ax.bar_label(bars, labels=[f"{v/1000:.1f}k" for v in values], padding=3, fontsize=7.5, rotation=0)
    ax.set_xticks(x, ["5 min", "15 min", "60 min"])
    ax.set_ylabel("MAE (tokens per 5-minute target window)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    ax.legend(ncol=4, loc="upper left")
    subtitle = "Validation-only model selection evidence" if split_name == "valid" else "Single post-freeze test evaluation; no test-informed retuning"
    title(ax, f"{code} {split_name.capitalize()} MAE by horizon", subtitle)
    footer(fig, "Source: table_02_baseline_results.csv, table_04_xgb_results.csv, and W4 LSTM outputs")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, filename)


def plot_representative_day(data: dict[str, object]) -> Path:
    frame = data["representative"]  # type: ignore[assignment]
    day = int(data["representative_day"])
    fig, ax = plt.subplots(figsize=(12.3, 6.1))
    ax.plot(frame["relative_hour"], frame["actual_token_load"], color=INK, linewidth=1.6, label="Actual")
    for model, column in {
        "Persistence": "persistence_prediction", "Seasonal Naive": "seasonal_naive_prediction",
        "XGBoost": "xgboost_prediction", "LSTM": "lstm_prediction",
    }.items():
        ax.plot(frame["relative_hour"], frame[column], color=MODEL_COLORS[model], linestyle=MODEL_LINES[model], linewidth=1.15 if model != "LSTM" else 1.4, label=model)
    ax.set_xlim(0, 24)
    ax.set_xlabel("Hour within the selected relative day")
    ax.set_ylabel("Tokens per 5-minute target window")
    ax.legend(ncol=5, loc="upper right")
    title(ax, "W4-06 15-minute forecasts on a representative test day", f"Complete relative day {day}, selected by median daily load among complete test days")
    footer(fig, "Source: pred_lstm_test_h15.csv, pred_xgb_test_h15.csv, and pred_baselines_test.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_06_h15_representative_day.png")


def plot_bias(data: dict[str, object]) -> Path:
    frame = data["bias"].copy()  # type: ignore[union-attr]
    frame["model_label"] = pd.Categorical(frame["model_label"], MODEL_ORDER[::-1], ordered=True)
    frame = frame.sort_values("model_label")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.hlines(y, frame["actual_mean"] / 1000, frame["prediction_mean"] / 1000, color=GRID, linewidth=3)
    ax.scatter(frame["actual_mean"] / 1000, y, color=INK, s=55, label="Actual mean")
    ax.scatter(frame["prediction_mean"] / 1000, y, color=[MODEL_COLORS[str(m)] for m in frame["model_label"]], marker="s", s=60, label="Prediction mean")
    for idx, row in enumerate(frame.itertuples()):
        ax.text(max(row.actual_mean, row.prediction_mean) / 1000 + 0.4, idx, f"bias {row.mean_error_bias/1000:+.1f}k", va="center", fontsize=8)
    ax.set_yticks(y, frame["model_label"].astype(str))
    ax.set_xlabel("Mean tokens per 5-minute target window (thousands)")
    ax.legend(loc="lower right")
    title(ax, "W4-07 15-minute test mean bias", "Same 5,228 target windows for all four methods; direction matters under the low-load test regime")
    footer(fig, "Source: aligned h15 prediction CSVs; test analysis is descriptive only")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_07_h15_test_bias.png")


def plot_burst_outcomes(data: dict[str, object]) -> Path:
    frame = data["metrics"]  # type: ignore[assignment]
    frame = frame.loc[frame["split"].eq("test") & frame["model_label"].isin(MODEL_ORDER)].copy()
    x = np.arange(3)
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    for i, model in enumerate(MODEL_ORDER):
        values = [int(frame.loc[frame["horizon_minutes"].eq(h) & frame["model_label"].eq(model), "predicted_burst_windows"].iloc[0]) for h in (5, 15, 60)]
        bars = ax.bar(x + (i - 1.5) * width, values, width, color=MODEL_COLORS[model], edgecolor=INK if model == "LSTM" else "white", linewidth=0.7, label=model)
        ax.bar_label(bars, labels=[str(v) for v in values], padding=2, fontsize=7.5)
    actual = int(frame["actual_burst_windows"].dropna().iloc[0])
    all_zero = bool(np.isclose(frame["f1"].astype(float), 0.0).all())
    ax.axhline(actual, color=INK, linestyle="--", linewidth=1.1, label=f"Actual count = {actual}")
    ax.set_xticks(x, ["5 min", "15 min", "60 min"])
    ax.set_ylabel("Predicted burst windows")
    ax.legend(ncol=5, loc="upper left")
    subtitle = (
        "Predicted counts do not imply hits: all reported test F1 values are zero"
        if all_zero
        else "Predicted counts do not imply hits; consult the metric table for precision, recall, and F1"
    )
    title(ax, "W4-08 Fixed-P95 burst outcomes", subtitle)
    footer(fig, "Source: one-time test metric tables | threshold fixed from training token-load P95")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_08_burst_outcomes.png")


def plot_training_profile(data: dict[str, object]) -> Path:
    frame = data["selected"].sort_values("horizon_minutes")  # type: ignore[union-attr]
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    ax.hlines(y, frame["train_mae_tokens"] / 1000, frame["valid_mae_tokens"] / 1000, color=GRID, linewidth=4)
    ax.scatter(frame["train_mae_tokens"] / 1000, y, color=BLUE, s=65, label="Train MAE")
    ax.scatter(frame["valid_mae_tokens"] / 1000, y, color=GOLD, marker="s", s=65, label="Validation MAE")
    for idx, row in enumerate(frame.iloc[::-1].itertuples()):
        ax.text(max(row.train_mae_tokens, row.valid_mae_tokens) / 1000 + 0.6, idx, f"best epoch {int(row.best_epoch)}", va="center", fontsize=8)
    ax.set_yticks(y, [f"h{int(h)}" for h in frame["horizon_minutes"]])
    ax.set_xlabel("MAE (thousands of tokens per target window)")
    right_edge = max(float(frame["train_mae_tokens"].max()), float(frame["valid_mae_tokens"].max())) / 1000
    left_edge = min(float(frame["train_mae_tokens"].min()), float(frame["valid_mae_tokens"].min())) / 1000
    ax.set_xlim(left_edge - 1.3, right_edge + 5.5)
    ax.legend(loc="upper right")
    title(ax, "W4-09 Frozen LSTM training profile", "One 5,537-parameter LSTM(32) candidate per horizon; final refit uses the selected best epoch")
    footer(fig, "Source: lstm_tuning_log.csv | selection_data = validation only")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w4_py_09_training_profile.png")


def create_overview(paths: list[Path]) -> Path:
    thumbs: list[Image.Image] = []
    cell = (720, 520)
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(cell, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", cell, "white")
            canvas.paste(image, ((cell[0] - image.width) // 2, (cell[1] - image.height) // 2))
            thumbs.append(ImageOps.expand(canvas, border=2, fill=GRID))
    overview = Image.new("RGB", (cell[0] * 3, cell[1] * 3), "white")
    for index, image in enumerate(thumbs):
        overview.paste(image, ((index % 3) * cell[0], (index // 3) * cell[1]))
    path = FIGURES / "w4_python_overview.png"
    overview.save(path, dpi=(180, 180))
    return path


def main() -> None:
    configure_style()
    data = export_shared_data()
    paths = [
        plot_sequence_audit(data),
        plot_feature_composition(data),
        plot_training_history(data),
        plot_mae(data, "valid", "W4-04", "fig_w4_py_04_validation_mae.png"),
        plot_mae(data, "test", "W4-05", "fig_w4_py_05_test_mae.png"),
        plot_representative_day(data),
        plot_bias(data),
        plot_burst_outcomes(data),
        plot_training_profile(data),
    ]
    create_overview(paths)
    manifest = pd.DataFrame(
        {"figure_number": [f"W4-{i:02d}" for i in range(1, 10)], "python_file": [p.name for p in paths], "bytes": [p.stat().st_size for p in paths]}
    )
    manifest.to_csv(VIZ_TABLES / "w4_python_figure_manifest.csv", index=False)
    print(f"Saved {len(paths)} Python W4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
