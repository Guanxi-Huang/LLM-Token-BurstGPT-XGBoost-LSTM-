"""Create the complete Week-1 visualization bundle with Python.

The script treats the cleaned Parquet files and the saved W1 audit tables as
the source of truth.  Besides the Python figures, it exports compact plotting
CSVs under ``outputs/tables/w1_visualization`` for the companion RStudio
script, so both toolchains use identical numbers.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_CONFIG_DIR = _PROJECT_ROOT / "outputs" / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pyarrow.parquet as pq


ROOT = _PROJECT_ROOT
TABLES = ROOT / "outputs" / "tables"
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures" / "w1_python"
VIZ_TABLES = TABLES / "w1_visualization"

BLUE = "#2F6B9A"
BLUE_LIGHT = "#BFD7EA"
GOLD = "#D9A441"
ORANGE = "#D97706"
OLIVE = "#728C3B"
PINK = "#C65A78"
INK = "#222222"
MID = "#6B7280"
GRID = "#D9DEE5"
OPEN = "#EFF3F7"

BATCH_COLORS = {"Batch 1": BLUE, "Batch 2": GOLD, "Batch 3": ORANGE}
MODEL_COLORS = {
    "persistence": MID,
    "seasonal_naive": GOLD,
    "xgboost": BLUE,
}


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
            "font.size": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def title(ax: plt.Axes, heading: str, subtitle: str) -> None:
    ax.set_title(heading, loc="left", fontsize=14, fontweight="bold", pad=30)
    ax.text(
        0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MID,
    )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=8, color=MID)


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_and_export_inputs() -> dict[str, pd.DataFrame]:
    quality = pd.read_csv(TABLES / "table_00_data_quality.csv")
    batches = quality.loc[quality["file_id"].str.startswith("without_fails")].copy()
    batches["batch"] = "Batch " + batches["file_id"].str.extract(r"(\d+)$")[0]
    batches["deletion_pct"] = 100 * batches["deletion_ratio"]
    quality_export = batches[
        [
            "batch",
            "raw_rows",
            "clean_rows",
            "deleted_rows",
            "deletion_pct",
            "max_clean_total_tokens",
            "timestamp_min_relative_seconds",
            "timestamp_max_relative_seconds",
            "duration_days_inclusive",
            "gaps_over_one_hour_in_file_order",
        ]
    ].copy()
    quality_export.to_csv(VIZ_TABLES / "w1_quality_summary.csv", index=False)

    daily_parts: list[pd.DataFrame] = []
    histogram_parts: list[pd.DataFrame] = []
    log_edges = np.linspace(0, 11.0, 90)
    parquet_paths = {
        "Batch 1": DATA / "requests_1.parquet",
        "Batch 2": DATA / "requests_2.parquet",
    }
    for batch, path in parquet_paths.items():
        parquet = pq.ParquetFile(path)
        day_frames: list[pd.DataFrame] = []
        histogram_counts = np.zeros(len(log_edges) - 1, dtype="int64")
        for record_batch in parquet.iter_batches(
            batch_size=200_000,
            columns=["timestamp_relative_seconds", "total_tokens"],
        ):
            frame = record_batch.to_pandas()
            frame["relative_day"] = (
                frame["timestamp_relative_seconds"] // 86_400
            ).astype("int64")
            day_frames.append(
                frame.groupby("relative_day", as_index=False).agg(
                    request_count=("total_tokens", "size"),
                    total_tokens=("total_tokens", "sum"),
                )
            )
            chunk_counts, _ = np.histogram(
                np.log1p(frame["total_tokens"].to_numpy(dtype="float64")),
                bins=log_edges,
            )
            histogram_counts += chunk_counts

        daily = (
            pd.concat(day_frames, ignore_index=True)
            .groupby("relative_day", as_index=False)
            .sum(numeric_only=True)
        )
        daily["batch"] = batch
        daily_parts.append(daily)
        histogram_parts.append(
            pd.DataFrame(
                {
                    "batch": batch,
                    "log_bin_left": log_edges[:-1],
                    "log_bin_right": log_edges[1:],
                    "log_bin_mid": (log_edges[:-1] + log_edges[1:]) / 2,
                    "request_count": histogram_counts,
                    "request_share": histogram_counts / histogram_counts.sum(),
                }
            )
        )

    daily_requests = pd.concat(daily_parts, ignore_index=True)
    token_histogram = pd.concat(histogram_parts, ignore_index=True)
    daily_requests.to_csv(VIZ_TABLES / "w1_daily_requests.csv", index=False)
    token_histogram.to_csv(VIZ_TABLES / "w1_token_histogram.csv", index=False)

    composition_rows: list[dict[str, object]] = []
    for row in batches.itertuples(index=False):
        for dimension, encoded in (
            ("Model", row.model_frequencies),
            ("Log type", row.log_type_frequencies),
        ):
            values = json.loads(encoded)
            denominator = sum(values.values())
            for category, count in values.items():
                composition_rows.append(
                    {
                        "batch": row.batch,
                        "dimension": dimension,
                        "category": category,
                        "count": count,
                        "share": count / denominator,
                    }
                )
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(VIZ_TABLES / "w1_category_composition.csv", index=False)

    field_rows: list[dict[str, object]] = []
    field_support = {
        "Relative timestamp": [1, 1, 1],
        "Request tokens": [1, 1, 1],
        "Response tokens": [1, 1, 1],
        "Reported total tokens": [1, 1, 1],
        "Model": [1, 1, 1],
        "Log type / call mode": [1, 1, 1],
        "Session ID": [0, 0, 0.5],
        "Elapsed time": [0, 0, 0.5],
        "Independent call-method field": [0, 0, 0],
        "Calendar date / weekday": [0, 0, 0],
    }
    for field, values in field_support.items():
        for batch_number, support in enumerate(values, start=1):
            field_rows.append(
                {
                    "field": field,
                    "batch": f"Batch {batch_number}",
                    "support": support,
                    "status": {0: "Not provided", 0.5: "Partial / audit only", 1: "Provided"}[
                        support
                    ],
                }
            )
    field_matrix = pd.DataFrame(field_rows)
    field_matrix.to_csv(VIZ_TABLES / "w1_field_support.csv", index=False)

    literature_map = {
        "[1]": [1, 1, 1, 0, 0, 0],
        "[2]": [1, 1, 1, 0, 0, 0],
        "[3]": [1, 1, 1, 1, 0, 0],
        "[4]": [0, 0, 1, 1, 0, 0],
        "[5]": [0, 0, 1, 1, 0, 0],
        "[6]": [0, 0, 1, 1, 0, 0],
        "[7]": [0, 0, 0, 1, 0, 0],
        "[8]": [0, 0, 1, 1, 0, 0],
        "[9]": [1, 0, 1, 1, 0, 0],
        "[10]": [0, 1, 0, 0, 1, 0],
        "[11]": [0, 1, 0, 0, 1, 0],
        "[15]": [0, 1, 0, 0, 0, 1],
    }
    themes = [
        "Workload characterization",
        "Forecasting",
        "Burst / SLO",
        "Serving systems",
        "Algorithm foundation",
        "Baseline / evaluation",
    ]
    literature_rows = [
        {"paper": paper, "theme": theme, "covered": values[index]}
        for paper, values in literature_map.items()
        for index, theme in enumerate(themes)
    ]
    literature = pd.DataFrame(literature_rows)
    literature.to_csv(VIZ_TABLES / "w1_literature_theme_matrix.csv", index=False)

    smoke = pd.read_csv(DATA / "smoke_2days_5min.csv")
    smoke["relative_hour"] = smoke["bin_start_relative_seconds"] / 3600
    smoke.to_csv(VIZ_TABLES / "w1_smoke_2days.csv", index=False)

    metrics_path = TABLES / "preliminary_cleaned_model_metrics_h15.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(VIZ_TABLES / "w1_preliminary_model_metrics.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "available",
        "data_policy": "Cleaned Parquet for batches 1/2; batch 3 audit summary only",
    }
    (VIZ_TABLES / "python_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return {
        "quality": quality,
        "batches": batches,
        "quality_export": quality_export,
        "daily": daily_requests,
        "histogram": token_histogram,
        "composition": composition,
        "fields": field_matrix,
        "literature": literature,
        "smoke": smoke,
        "metrics": metrics,
    }


def plot_cleaning_retention(data: dict[str, pd.DataFrame]) -> Path:
    frame = data["quality_export"]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - 0.18, frame["raw_rows"] / 1e6, 0.36, color=BLUE_LIGHT, label="Raw")
    ax.bar(x + 0.18, frame["clean_rows"] / 1e6, 0.36, color=BLUE, label="Cleaned")
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.text(
            index + 0.18,
            row["clean_rows"] / 1e6 + 0.08,
            f"-{row['deletion_pct']:.2f}%",
            ha="center",
            fontsize=9,
            color=INK,
        )
    failure = data["quality"].loc[
        data["quality"]["file_id"].eq("complete_1"), "zero_output_token_ratio"
    ].iloc[0]
    ax.text(
        0.99,
        0.82,
        f"complete_1 failure audit\nzero output tokens: {failure:.2%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": OPEN, "edgecolor": GRID},
    )
    ax.set_xticks(x, frame["batch"])
    ax.set_ylabel("Requests (millions)")
    ax.set_ylim(0, frame["raw_rows"].max() / 1e6 * 1.18)
    ax.legend(ncol=2, loc="upper left")
    title(
        ax,
        "W1-01 Cleaning retention by batch",
        "All sequential deletion counts; exact all-field duplicates were the only removed records",
    )
    footer(fig, "Source: table_00_data_quality.csv and table_00_cleaning_steps.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_01_cleaning_retention.png")


def plot_field_support(data: dict[str, pd.DataFrame]) -> Path:
    fields = data["fields"]
    field_order = fields["field"].drop_duplicates().tolist()
    batch_order = ["Batch 1", "Batch 2", "Batch 3"]
    matrix = (
        fields.pivot(index="field", columns="batch", values="support")
        .reindex(index=field_order, columns=batch_order)
        .to_numpy()
    )
    cmap = ListedColormap(["#E7E9ED", GOLD, BLUE])
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    labels = {0: "No", 0.5: "Partial", 1: "Yes"}
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                labels[matrix[row, column]],
                ha="center",
                va="center",
                color="white" if matrix[row, column] == 1 else INK,
                fontsize=8.5,
                fontweight="bold" if matrix[row, column] == 1 else "normal",
            )
    ax.set_xticks(range(len(batch_order)), batch_order)
    ax.set_yticks(range(len(field_order)), field_order)
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Not provided", "Partial / audit only", "Provided"])
    title(
        ax,
        "W1-02 Source-field availability",
        "Batch 1/2 do not support Session, elapsed-time, calendar, or a separate call-method feature",
    )
    footer(fig, "Source: sampled CSV schemas and docs/data_dictionary.md")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_02_field_support.png")


def plot_batch_timeline(data: dict[str, pd.DataFrame]) -> Path:
    frame = data["quality_export"].copy()
    frame["start_day"] = frame["timestamp_min_relative_seconds"] / 86_400
    frame["end_day"] = frame["timestamp_max_relative_seconds"] / 86_400
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y_positions = np.arange(len(frame))[::-1]
    for y, row in zip(y_positions, frame.itertuples(index=False)):
        ax.hlines(
            y,
            row.start_day,
            row.end_day,
            linewidth=16,
            color=BATCH_COLORS[row.batch],
            alpha=0.9,
        )
        ax.text(
            row.end_day + 2,
            y,
            f"{row.duration_days_inclusive:.1f} days",
            va="center",
            fontsize=9,
        )
    gap_start = frame.loc[frame["batch"].eq("Batch 2"), "end_day"].iloc[0]
    gap_end = frame.loc[frame["batch"].eq("Batch 3"), "start_day"].iloc[0]
    ax.axvspan(gap_start, gap_end, color=OPEN, zorder=-1)
    ax.text(
        (gap_start + gap_end) / 2,
        -0.55,
        f"Cross-period gap: {gap_end - gap_start:.0f} days",
        ha="center",
        color=MID,
        fontsize=9,
    )
    ax.set_yticks(y_positions, frame["batch"])
    ax.set_xlabel("Relative day (collection calendar undisclosed)")
    ax.set_ylim(-0.8, len(frame) - 0.25)
    title(
        ax,
        "W1-03 Batch time coverage",
        "Batches 1/2 are contiguous for the main experiment; Batch 3 is reserved for robustness testing",
    )
    footer(fig, "Source: table_00_data_quality.csv; synthetic UTC dates are not calendar dates")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_03_batch_time_coverage.png")


def plot_daily_requests(data: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for batch, group in data["daily"].groupby("batch", sort=False):
        ax.plot(
            group["relative_day"],
            group["request_count"] / 1000,
            color=BATCH_COLORS[batch],
            linewidth=1.8,
            label=batch,
        )
    ax.set_xlabel("Relative day")
    ax.set_ylabel("Cleaned requests (thousands/day)")
    ax.legend(ncol=2, loc="upper right")
    title(
        ax,
        "W1-04 Daily request volume",
        "Cleaned main-experiment batches only; large level changes confirm non-stationary traffic",
    )
    footer(fig, "Source: requests_1.parquet and requests_2.parquet")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_04_daily_requests.png")


def plot_token_distribution(data: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for batch, group in data["histogram"].groupby("batch", sort=False):
        token_mid = np.expm1(group["log_bin_mid"])
        positive = group["request_share"] > 0
        ax.step(
            token_mid[positive],
            group.loc[positive, "request_share"],
            where="mid",
            linewidth=1.8,
            color=BATCH_COLORS[batch],
            label=batch,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total tokens per request (log scale)")
    ax.set_ylabel("Share of requests per bin (log scale)")
    ax.legend(ncol=2, loc="upper right")
    title(
        ax,
        "W1-05 Request-level total-token distribution",
        "Full cleaned Batch 1/2 distributions; both are long-tailed and materially different",
    )
    footer(fig, "Source: full requests_1.parquet and requests_2.parquet; 89 equal-width log bins")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_05_token_distribution.png")


def plot_composition(data: dict[str, pd.DataFrame]) -> Path:
    composition = data["composition"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True)
    palette = {
        "ChatGPT": BLUE,
        "GPT-4": GOLD,
        "API log": BLUE,
        "Conversation log": ORANGE,
    }
    for ax, dimension in zip(axes, ["Model", "Log type"]):
        subset = composition.loc[composition["dimension"].eq(dimension)]
        pivot = subset.pivot(index="batch", columns="category", values="share").fillna(0)
        pivot = pivot.reindex(["Batch 1", "Batch 2", "Batch 3"])
        left = np.zeros(len(pivot))
        for category in pivot.columns:
            values = pivot[category].to_numpy()
            ax.barh(
                pivot.index,
                values,
                left=left,
                color=palette.get(category, MID),
                label=category,
            )
            for index, (start, value) in enumerate(zip(left, values)):
                if value >= 0.06:
                    ax.text(
                        start + value / 2,
                        index,
                        f"{value:.1%}",
                        ha="center",
                        va="center",
                        color="white" if category in {"ChatGPT", "API log"} else INK,
                        fontsize=8,
                    )
            left += values
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of raw requests")
        ax.set_title(dimension, fontsize=11, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(
        "W1-06 Model and log-type composition",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.91,
        "Raw audit composition; Batch 2 is substantially more concentrated in ChatGPT and API logs",
        fontsize=9,
        color=MID,
    )
    footer(fig, "Source: model_frequencies and log_type_frequencies in table_00_data_quality.csv")
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    return save(fig, "fig_w1_py_06_category_composition.png")


def plot_smoke_profile(data: dict[str, pd.DataFrame]) -> Path:
    smoke = data["smoke"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    axes[0].plot(smoke["relative_hour"], smoke["total_tokens"], color=BLUE, linewidth=1.1)
    axes[0].set_ylabel("Total tokens / 5 min")
    axes[0].set_title("Token load", loc="left", fontsize=10, fontweight="bold")
    axes[1].plot(smoke["relative_hour"], smoke["request_count"], color=GOLD, linewidth=1.1)
    axes[1].set_ylabel("Requests / 5 min")
    axes[1].set_xlabel("Hours from smoke-test start")
    axes[1].set_title("Request count", loc="left", fontsize=10, fontweight="bold")
    fig.suptitle(
        "W1-07 Two-day end-to-end smoke-test profile",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.925,
        f"576 complete bins | {int(smoke['request_count'].sum()):,} requests | "
        f"{int(smoke['total_tokens'].sum()):,} tokens",
        fontsize=9,
        color=MID,
    )
    footer(fig, "Source: data/processed/smoke_2days_5min.csv")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    return save(fig, "fig_w1_py_07_smoke_profile.png")


def plot_literature_matrix(data: dict[str, pd.DataFrame]) -> Path:
    literature = data["literature"]
    paper_order = literature["paper"].drop_duplicates().tolist()
    theme_order = literature["theme"].drop_duplicates().tolist()
    matrix = (
        literature.pivot(index="paper", columns="theme", values="covered")
        .reindex(index=paper_order, columns=theme_order)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.imshow(matrix, cmap=ListedColormap([OPEN, BLUE]), vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if matrix[row, column]:
                ax.text(column, row, "●", ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(range(len(theme_order)), theme_order, rotation=30, ha="right")
    ax.set_yticks(range(len(paper_order)), paper_order)
    ax.grid(False)
    title(
        ax,
        "W1-08 Literature evidence themes",
        "Twelve priority papers coded from the completed literature matrix; marks show thematic coverage, not quality scores",
    )
    footer(fig, "Source: docs/literature_matrix.md; manual theme coding documented in this script")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_w1_py_08_literature_themes.png")


def plot_preliminary_models(data: dict[str, pd.DataFrame]) -> Path | None:
    metrics = data["metrics"]
    if metrics.empty:
        return None
    model_order = ["persistence", "seasonal_naive", "xgboost"]
    split_order = ["validation", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    width = 0.24
    x = np.arange(len(split_order))
    for index, model in enumerate(model_order):
        subset = metrics.loc[metrics["model"].eq(model)].set_index("split").reindex(split_order)
        offset = (index - 1) * width
        bars = axes[0].bar(
            x + offset,
            subset["mae"],
            width,
            color=MODEL_COLORS[model],
            label=model,
        )
        axes[0].bar_label(bars, labels=[f"{value/1000:.1f}k" for value in subset["mae"]], fontsize=8)
        bars_f1 = axes[1].bar(
            x + offset,
            subset["f1"],
            width,
            color=MODEL_COLORS[model],
            label=model,
        )
        axes[1].bar_label(bars_f1, labels=[f"{value:.3f}" for value in subset["f1"]], fontsize=8)
    axes[0].set_xticks(x, ["Validation", "Test"])
    axes[0].set_ylabel("MAE (tokens / 5 min)")
    axes[0].set_title("Continuous-load error", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x, ["Validation", "Test"])
    axes[1].set_ylabel("Burst F1 at fixed training P95")
    axes[1].set_ylim(0, max(0.55, metrics["f1"].max() * 1.25))
    axes[1].set_title("Fixed-threshold burst detection", fontsize=10, fontweight="bold")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.06, -0.16))
    fig.suptitle(
        "W1-09 Preliminary 15-minute-ahead model results",
        x=0.07,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.91,
        "Cleaned Parquet, chronological 70/15/15 split; test F1 is zero for all methods because the fixed threshold is sparse",
        fontsize=9,
        color=MID,
    )
    footer(fig, "Source: preliminary_cleaned_model_metrics_h15.csv; LSTM unavailable in W1")
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    return save(fig, "fig_w1_py_09_preliminary_models.png")


def make_overview(paths: list[Path]) -> Path:
    tiles: list[Image.Image] = []
    tile_width, tile_height = 760, 470
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width - 20, tile_height - 45), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image, ((tile_width - image.width) // 2, 32))
        ImageDraw.Draw(tile).text((12, 9), path.stem, fill=INK)
        tiles.append(tile)
    columns = 3
    rows = int(np.ceil(len(tiles) / columns))
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    output = FIGURES / "w1_python_overview.png"
    canvas.save(output, quality=95)
    return output


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ_TABLES.mkdir(parents=True, exist_ok=True)
    data = load_and_export_inputs()
    paths = [
        plot_cleaning_retention(data),
        plot_field_support(data),
        plot_batch_timeline(data),
        plot_daily_requests(data),
        plot_token_distribution(data),
        plot_composition(data),
        plot_smoke_profile(data),
        plot_literature_matrix(data),
    ]
    model_path = plot_preliminary_models(data)
    if model_path is not None:
        paths.append(model_path)
    overview = make_overview(paths)
    print(f"Created {len(paths)} Python figures and overview: {overview}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
