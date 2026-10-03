"""Create and execute the reader-facing Week-2 EDA notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_eda.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        markdown(
            """
            # BurstGPT 5-minute target-series EDA

            ## tl;dr

            This notebook analyses only the complete five-minute grid built from primary batches 1 and 2 and saves all figures to `outputs/figures/`. Public time is relative seconds, and the 1970 dates form only a synthetic UTC axis for resampling. The analysis therefore uses **relative time of day, relative weekly phase, and relative day number** without interpreting real weekdays, weekends, or holidays.
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import sys

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "processed" / "series_5min.parquet").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
            MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Image, Markdown, display
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from plot_style import (
                COLORS,
                HEATMAP_CMAP,
                MODEL_COLORS,
                MODEL_LINESTYLES,
                apply_plot_style,
                save_figure,
            )

            SEED = 42
            FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            BLUE = COLORS["blue"]
            BLUE_LIGHT = "#A9B9D6"
            GOLD = COLORS["orange"]
            ORANGE = COLORS["magenta"]
            INK = COLORS["ink"]
            GREY = COLORS["dark_gray"]
            apply_plot_style()

            def save_and_show(fig, filename):
                path = FIGURE_DIR / filename
                save_figure(fig, path)
                plt.close(fig)
                display(Image(filename=str(path)))
                return path

            series_path = PROJECT_ROOT / "data" / "processed" / "series_5min.parquet"
            series = pd.read_parquet(series_path)
            series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
            series.index.name = "timestamp"
            threshold_table = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "table_01_burst_threshold.csv")
            TRAIN_P95 = float(threshold_table.loc[0, "p95_threshold"])

            assert series.index.is_unique and series.index.is_monotonic_increasing
            assert series.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=5)).all()
            assert series.loc[series["request_count"].eq(0), ["mean_request_tokens", "mean_response_tokens"]].isna().all().all()

            zero_share = float(series["is_zero_window"].mean())
            load_cv = float(series["token_load"].std() / series["token_load"].mean())
            spearman = float(series[["request_count", "token_load"]].corr(method="spearman").iloc[0, 1])
            display(Markdown(
                f"**Executive summary.** The complete grid contains **{len(series):,}** windows, with **{zero_share:.2%}** zero-load windows. "
                f"The fixed training P95 is **{TRAIN_P95:,.2f} Token/5min**. The Spearman correlation between request count and token load is **{spearman:.3f}**, "
                f"showing that request volume is important but does not fully explain token load."
            ))
            """
        ),
        markdown(
            """
            ## Context & Methods

            ### Key Assumptions

            - Input: `series_5min.parquet`, aggregated by `src/02_build_series.py` from `data/processed/requests_1.parquet` and `requests_2.parquet`.
            - Windows: left-closed and right-open `[t, t+5min)`; zero-request windows are retained with zero load and count and NaN means.
            - Split and bursts: chronological 70%/15%/15%; P95 is calculated only on training data and fixed for all three splits.
            - Scope: feature design, naive baselines, and result interpretation; relative weekly phase is never interpreted as a real weekday or weekend.

            ## Data
            """
        ),
        code(
            """
            data_summary = pd.DataFrame({
                "value": [
                    series.index.min().isoformat(),
                    series.index.max().isoformat(),
                    len(series),
                    int(series["request_count"].sum()),
                    int(series["token_load"].sum()),
                    int(series["is_zero_window"].sum()),
                    TRAIN_P95,
                ]
            }, index=[
                "first synthetic UTC bin", "last synthetic UTC bin", "5-minute windows",
                "requests", "tokens", "zero windows", "train-only P95"
            ])
            display(data_summary)
            display(series.head(3))
            """
        ),
        markdown("## Results"),
        markdown("### Figure 01 — Complete 5-minute load trend"),
        code(
            """
            fig, ax = plt.subplots(figsize=(13, 5.2))
            relative_days = (series.index - series.index.min()).total_seconds() / 86_400
            ax.plot(relative_days, series["token_load"], color=BLUE_LIGHT, linewidth=1.8, alpha=0.20, label="5-minute load")
            rolling_day = series["token_load"].rolling(288, min_periods=1).mean()
            ax.plot(relative_days, rolling_day, color=BLUE, linewidth=2.2, label="24-hour trailing mean")
            ax.axhline(TRAIN_P95, color=GREY, linewidth=1.8, linestyle=":", label="Train P95")
            ax.set_xlabel("Relative day from trace start")
            ax.set_ylabel("Tokens per 5-minute window")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="upper right")
            save_and_show(fig, "fig_01_full_series_trend.png")
            first_half = rolling_day.iloc[:len(series)//2].median()
            second_half = rolling_day.iloc[len(series)//2:].median()
            display(Markdown(
                f"**Observation:** The load coefficient of variation is **{load_cv:.2f}** with clear long-tail spikes. The median 24-hour rolling mean in the second half is **{second_half / max(first_half, 1):.2f} times** the first-half value. "
                "This supports lagged and rolling features while warning that a fixed training threshold may become miscalibrated after distribution shift."
            ))
            """
        ),
        markdown("### Figure 02 — Representative relative week at 5-minute resolution"),
        code(
            """
            week_id = ((series.index - series.index.min()) // pd.Timedelta(days=7)).astype(int)
            week_stats = series.assign(relative_week=week_id).groupby("relative_week").agg(
                n_rows=("token_load", "size"), total_load=("token_load", "sum")
            )
            complete_weeks = week_stats.loc[week_stats["n_rows"].eq(7 * 288)]
            median_week_load = complete_weeks["total_load"].median()
            representative_week = int((complete_weeks["total_load"] - median_week_load).abs().idxmin())
            week_data = series.loc[week_id == representative_week]
            phase_days = np.arange(len(week_data)) / 288
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(phase_days, week_data["token_load"], color=BLUE, linewidth=1.8)
            ax.axhline(TRAIN_P95, color=GREY, linestyle=":", linewidth=1.8, label="Train P95")
            ax.set_xlabel("Relative day within selected 7-day phase (not calendar weekday)")
            ax.set_ylabel("Tokens per 5-minute window")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M"))
            ax.set_xlim(0, 7)
            ax.set_xticks(range(8))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            save_and_show(fig, "fig_02_representative_relative_week.png")
            peak_position = int(np.argmax(week_data["token_load"].to_numpy()))
            display(Markdown(
                f"**Observation:** A representative complete week still has strong window-to-window variation, with maximum load at relative weekly phase **{peak_position / 288:.2f} days**. "
                "This phase supports periodic features and the Seasonal Naive baseline but does not represent a real weekday."
            ))
            """
        ),
        markdown("### Figure 03 — Mean load by relative day-in-week and time-of-day phase"),
        code(
            """
            phase = pd.DataFrame(index=series.index)
            elapsed_bins = np.arange(len(series))
            phase["relative_day_in_week"] = (elapsed_bins // 288) % 7
            phase["slot_in_day"] = elapsed_bins % 288
            phase["token_load"] = series["token_load"].to_numpy()
            heatmap = phase.pivot_table(index="relative_day_in_week", columns="slot_in_day", values="token_load", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(14, 4.8))
            sns.heatmap(heatmap, cmap=HEATMAP_CMAP, ax=ax, cbar_kws={"label": "Mean tokens / 5 min"})
            ticks = np.arange(0, 289, 48)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(value // 12):02d}:00" for value in ticks])
            ax.set_yticklabels([f"Relative day {value + 1}" for value in heatmap.index], rotation=0)
            ax.set_xlabel("Relative time-of-day phase")
            ax.set_ylabel("Relative day within 7-day phase")
            save_and_show(fig, "fig_03_relative_week_phase_heatmap.png")
            peak_day, peak_slot = np.unravel_index(np.nanargmax(heatmap.to_numpy()), heatmap.shape)
            display(Markdown(
                f"**Observation:** The highest mean load occurs on relative day **{peak_day + 1}** at about **{peak_slot // 12:02d}:{(peak_slot % 12) * 5:02d}**. "
                "This phase structure supports sine and cosine features but cannot establish real weekday or weekend effects."
            ))
            """
        ),
        markdown("### Figure 04 — Log-transformed token-load distribution"),
        code(
            """
            positive_load = series.loc[series["token_load"].gt(0), "token_load"]
            positive_log_load = np.log1p(positive_load.to_numpy(dtype=float))
            zero_windows = int(series["token_load"].eq(0).sum())
            excluded_below_view = int((positive_log_load < 4).sum())
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(positive_log_load, bins=80, stat="density", color=BLUE, edgecolor="white", linewidth=0.25, ax=ax)
            ax.set_xlabel("log(1 + tokens per 5-minute window)")
            ax.set_ylabel("Density")
            ax.set_xlim(4, 16)
            ax.set_ylim(0.0, 0.3)
            ax.set_xticks(np.arange(4, 17, 2))
            ax.set_yticks(np.arange(0.0, 0.31, 0.05))
            ax.spines[["top", "right"]].set_visible(False)
            save_and_show(fig, "fig_04_token_load_log_distribution.png")
            q50, q95, q99 = series["token_load"].quantile([0.50, 0.95, 0.99]).to_numpy()
            display(Markdown(
                f"**Observation:** Median, P95, and P99 on the original scale are **{q50:,.0f} / {q95:,.0f} / {q99:,.0f} tokens**, with **{zero_share:.2%}** zero windows. "
                "The long tail supports comparing original and log1p training targets, while evaluation should remain on the original token scale."
            ))
            """
        ),
        markdown("### Figure 05 — Request count versus token load"),
        code(
            """
            scatter_data = series[["request_count", "token_load", "split"]]
            if len(scatter_data) > 20_000:
                scatter_data = scatter_data.sample(20_000, random_state=SEED)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(
                np.log1p(scatter_data["request_count"]),
                np.log1p(scatter_data["token_load"]),
                s=10, alpha=0.22, color=BLUE, edgecolors="none"
            )
            ax.set_xlabel("log(1 + requests per 5-minute window)")
            ax.set_ylabel("log(1 + tokens per 5-minute window)")
            ax.spines[["top", "right"]].set_visible(False)
            save_and_show(fig, "fig_05_request_count_vs_token_load.png")
            display(Markdown(
                f"**Observation:** Request count has a strong monotonic relationship with token load (Spearman **{spearman:.3f}**), but the same request count still spans a wide load range. "
                "Historical request count is therefore useful, while mean input/output tokens and model composition can explain variation in request size."
            ))
            """
        ),
        markdown("### Figure 06 — Model request mix over relative time"),
        code(
            """
            daily = series[["request_count", "gpt4_request_count", "chatgpt_request_count", "api_request_count", "conversation_request_count"]].resample("1D").sum()
            daily["gpt4_share"] = daily["gpt4_request_count"].div(daily["request_count"].replace(0, np.nan))
            daily["chatgpt_share"] = daily["chatgpt_request_count"].div(daily["request_count"].replace(0, np.nan))
            daily_relative_day = np.arange(len(daily))
            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.plot(daily_relative_day, daily["gpt4_share"], color=BLUE, linewidth=2.0, label="GPT-4")
            ax.plot(daily_relative_day, daily["chatgpt_share"], color=GOLD, linewidth=1.8, linestyle="--", label="ChatGPT")
            ax.set_xlabel("Relative day")
            ax.set_ylabel("Share of requests")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            save_and_show(fig, "fig_06_model_share_over_time.png")
            gpt4_min, gpt4_max = daily["gpt4_share"].min(), daily["gpt4_share"].max()
            display(Markdown(
                f"**Observation:** Daily GPT-4 request share ranges from **{gpt4_min:.2%} to {gpt4_max:.2%}**, so model composition is not constant. "
                "Prediction features may use only composition statistics available before forecast time, never the actual future-window model share."
            ))
            """
        ),
        markdown("### Figure 07 — Log Type request mix over relative time"),
        code(
            """
            daily["api_share"] = daily["api_request_count"].div(daily["request_count"].replace(0, np.nan))
            daily["conversation_share"] = daily["conversation_request_count"].div(daily["request_count"].replace(0, np.nan))
            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.plot(daily_relative_day, daily["api_share"], color=BLUE, linewidth=2.0, label="API log")
            ax.plot(daily_relative_day, daily["conversation_share"], color=ORANGE, linewidth=1.8, linestyle="--", label="Conversation log")
            ax.set_xlabel("Relative day")
            ax.set_ylabel("Share of requests")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            save_and_show(fig, "fig_07_log_type_share_over_time.png")
            api_min, api_max = daily["api_share"].min(), daily["api_share"].max()
            display(Markdown(
                f"**Observation:** Daily API log share ranges from **{api_min:.2%} to {api_max:.2%}**, and composition differences between batches create temporal drift. "
                "Historical API share is a valid lagged feature, while future-window Log Type composition cannot be a direct model input."
            ))
            """
        ),
        markdown("### Figure 08 — Autocorrelation of log load"),
        code(
            """
            transformed_load = pd.Series(np.log1p(series["token_load"].to_numpy(dtype=float)), index=series.index)
            fig, ax = plt.subplots(figsize=(12, 4.8))
            plot_acf(transformed_load, lags=288, zero=False, fft=True, ax=ax, title="", color=BLUE, vlines_kwargs={"colors": BLUE, "linewidth": 1.8})
            ax.set_xlabel("Lag (5-minute steps)")
            ax.set_ylabel("ACF")
            ax.spines[["top", "right"]].set_visible(False)
            save_and_show(fig, "fig_08_acf_log_load.png")
            acf_values = [transformed_load.autocorr(lag=value) for value in (1, 3, 12, 288)]
            display(Markdown(
                f"**Observation:** Log-load ACF at 1, 3, 12, and 288 steps is **{acf_values[0]:.3f}/{acf_values[1]:.3f}/{acf_values[2]:.3f}/{acf_values[3]:.3f}**. "
                "Short-term persistence and the 288-step daily cycle should enter the candidate lag set, with gains compared on validation data."
            ))
            """
        ),
        markdown("### Figure 09 — Partial autocorrelation of log load"),
        code(
            """
            fig, ax = plt.subplots(figsize=(12, 4.8))
            plot_pacf(transformed_load, lags=288, zero=False, method="ywm", ax=ax, title="", color=BLUE, vlines_kwargs={"colors": BLUE, "linewidth": 1.8})
            ax.set_xlabel("Lag (5-minute steps)")
            ax.set_ylabel("PACF")
            ax.spines[["top", "right"]].set_visible(False)
            save_and_show(fig, "fig_09_pacf_log_load.png")
            display(Markdown(
                "**Observation:** After controlling for intermediate lags, significance is concentrated in short lags near the origin, while several distant phases retain signal. "
                "This supports sparse short, hourly, and 288-step seasonal lags instead of every consecutive lag."
            ))
            """
        ),
        markdown("### Figure 10 — Fifteen-minute baselines on a representative test day"),
        code(
            """
            baseline = pd.read_csv(
                PROJECT_ROOT / "outputs" / "tables" / "pred_baselines_test.csv",
                parse_dates=["timestamp"],
            )
            baseline["timestamp"] = pd.to_datetime(baseline["timestamp"], utc=True)
            baseline["relative_test_day"] = (
                (baseline["timestamp"] - baseline["timestamp"].min())
                // pd.Timedelta(days=1)
            ).astype(int)
            day_stats = baseline.groupby("relative_test_day").agg(
                n_rows=("actual_token_load", "size"),
                total_load=("actual_token_load", "sum"),
            )
            complete_days = day_stats.loc[day_stats["n_rows"].eq(288)]
            representative_day = int(
                (complete_days["total_load"] - complete_days["total_load"].median())
                .abs()
                .idxmin()
            )
            representative = baseline.loc[
                baseline["relative_test_day"].eq(representative_day)
            ].copy()
            phase_hours = np.arange(len(representative)) * 5 / 60

            fig, ax = plt.subplots(figsize=(13, 5.2))
            ax.plot(
                phase_hours,
                representative["actual_token_load"],
                color=MODEL_COLORS["Actual"],
                linewidth=1.8,
                label="Actual",
            )
            ax.plot(
                phase_hours,
                representative["pred_persistence_h15"],
                color=MODEL_COLORS["Persistence"],
                linewidth=1.8,
                linestyle=MODEL_LINESTYLES["Persistence"],
                label="Persistence (last completed window)",
            )
            ax.plot(
                phase_hours,
                representative["pred_seasonal_h15"],
                color=MODEL_COLORS["Seasonal Naive"],
                linewidth=1.8,
                linestyle=MODEL_LINESTYLES["Seasonal Naive"],
                label="Seasonal naive (target minus 24 h)",
            )
            ax.set_xlabel("Relative hour within selected test day")
            ax.set_ylabel("Token load (Token/5min)")
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 4))
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda value, _: f"{value / 1e3:.0f}k"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            save_and_show(fig, "fig_10_baseline_representative_day.png")
            persistence_mae = float(
                np.mean(
                    np.abs(
                        representative["pred_persistence_h15"]
                        - representative["actual_token_load"]
                    )
                )
            )
            seasonal_mae = float(
                np.mean(
                    np.abs(
                        representative["pred_seasonal_h15"]
                        - representative["actual_token_load"]
                    )
                )
            )
            display(Markdown(
                f"**Observation:** On this objectively selected complete test day, 15-minute MAE for Persistence and Seasonal Naive is "
                f"**{persistence_mae:,.0f}/{seasonal_mae:,.0f} Token/5min**。"
                "The figure shows window-level tracking on a representative day, while final model comparison still uses all 5,228 test target windows."
            ))
            """
        ),
        markdown("## Takeaways"),
        code(
            """
            display(Markdown(
                f'''
                1. Preserve the complete grid and zero windows: the series has **{len(series):,}** five-minute windows, including **{int(series['is_zero_window'].sum()):,}** zero-load windows. Mean features remain NaN in these windows and may only be filled from history.
                2. Short-term persistence, hourly structure, and the 288-step relative daily cycle have statistical support, so Persistence and Seasonal Naive are required baselines.
                3. Request count is strongly related to load but does not fully explain token size. Historical means, model composition, and Log Type composition are valid candidate features only after lagging to prevent future leakage.
                4. The fixed training P95 is **{TRAIN_P95:,.2f} Token/5min**. Later load and composition drift means burst results must report the cross-period limitation of a fixed threshold.
                '''
            ))
            """
        ),
    ]
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()
    nbf.write(executed, NOTEBOOK_PATH)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    print(f"Created and executed {NOTEBOOK_PATH} ({len(code_cells)} code cells, no errors)")


if __name__ == "__main__":
    main()
