from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def one_value(frame: pd.DataFrame, **filters: object) -> float:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return float(selected.iloc[0]["mae"])


def main() -> None:
    results = pd.read_csv(TABLES / "table_04_xgb_results.csv")
    comparison = pd.read_csv(TABLES / "table_04_xgb_vs_baselines_h15.csv")
    values = [
        one_value(
            results,
            split="valid",
            model="xgboost_default",
            horizon_minutes=15,
        ),
        one_value(
            results,
            split="valid",
            model="xgboost_selected",
            horizon_minutes=15,
        ),
        float(
            comparison.loc[comparison["baseline"].eq("persistence"), "baseline_mae"].iloc[0]
        ),
        float(
            comparison.loc[comparison["baseline"].eq("seasonal_naive"), "baseline_mae"].iloc[0]
        ),
        one_value(
            results,
            split="test",
            model="xgboost_final",
            horizon_minutes=15,
        ),
    ]
    labels = [
        "Validation\ndefault XGBoost",
        "Validation\nselected XGBoost",
        "Test\nPersistence",
        "Test\nSeasonal Naive",
        "Test\nfrozen XGBoost",
    ]
    colors = ["#5B5B5B", "#2F5597", "#5B5B5B", "#D97706", "#2F5597"]
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    bars = axis.bar(range(len(values)), values, color=colors, alpha=0.9)
    axis.axvline(1.5, color="#B8B8B8", linewidth=1.2, linestyle="--")
    axis.set_ylabel("MAE (Token/5min)")
    axis.set_xticks(range(len(labels)), labels)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#B8B8B8", alpha=0.2)
    axis.set_axisbelow(True)
    axis.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=4, fontsize=9)
    axis.set_ylim(0, max(values) * 1.18)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / "fig_w3_xgb_mae_evidence.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(FIGURES / "fig_w3_xgb_mae_evidence.png")


if __name__ == "__main__":
    main()
