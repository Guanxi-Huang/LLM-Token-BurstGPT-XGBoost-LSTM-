"""Validate the final Week-5 Python/RStudio figure bundles and shared data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "outputs" / "tables" / "w5_visualization"
PY_DIR = ROOT / "outputs" / "figures" / "w5_python"
R_DIR = ROOT / "outputs" / "figures" / "w5_rstudio"
OUTPUT = ROOT / "outputs" / "tables" / "table_09_w5_visualization_validation.csv"
MODELS = {"Persistence", "Seasonal Naive", "XGBoost", "LSTM"}


def check_image(tool: str, figure_id: str, path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "toolchain": tool,
        "figure_id": figure_id,
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "width_px": 0,
        "height_px": 0,
        "grayscale_std": 0.0,
        "passed": False,
        "reason": "missing",
    }
    if not path.exists():
        return result
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        result["width_px"], result["height_px"] = rgb.size
        preview = rgb.copy()
        preview.thumbnail((800, 800))
        stat = ImageStat.Stat(preview.convert("L"))
        result["grayscale_std"] = float(stat.stddev[0])
    conditions = {
        "bytes": int(result["bytes"]) >= 15_000,
        "dimensions": int(result["width_px"]) >= 1_200 and int(result["height_px"]) >= 700,
        "content": float(result["grayscale_std"]) >= 5.0,
    }
    failed = [name for name, value in conditions.items() if not value]
    result["passed"] = not failed
    result["reason"] = "ok" if not failed else ",".join(failed)
    return result


def main() -> None:
    metrics_path = SHARED / "w5_metrics_all.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run src/12_visualize_w5_python.py first")
    metrics = pd.read_csv(metrics_path)
    if set(metrics["model"].astype(str)) != MODELS:
        raise AssertionError("Shared metrics do not contain exactly four frozen models")
    if set(metrics["horizon_minutes"].astype(int)) != {5, 15, 60}:
        raise AssertionError("Shared metrics do not contain all three horizons")
    coverage = metrics.groupby("horizon_minutes")["model"].nunique()
    if not coverage.eq(4).all() or len(metrics) != 12:
        raise AssertionError("Shared metrics are not a complete 3 x 4 grid")

    py_manifest = pd.read_csv(SHARED / "w5_python_figure_manifest.csv")
    r_manifest = pd.read_csv(SHARED / "w5_r_figure_manifest.csv")
    if len(py_manifest) != 10 or len(r_manifest) != 10:
        raise AssertionError("Each toolchain must contain exactly ten numbered figures")

    rows: list[dict[str, object]] = []
    for index, row in py_manifest.iterrows():
        rows.append(check_image("Python", f"W5-{index + 1:02d}", PY_DIR / str(row["python_file"])))
    for index, row in r_manifest.iterrows():
        rows.append(check_image("RStudio", f"W5-{index + 1:02d}", R_DIR / str(row["rstudio_file"])))
    for tool, path in (("Python", PY_DIR / "w5_python_overview.png"), ("RStudio", R_DIR / "w5_rstudio_overview.png")):
        rows.append(check_image(tool, "overview", path))

    audit = pd.DataFrame(rows)
    audit["shared_metric_rows"] = len(metrics)
    audit["shared_model_count"] = metrics["model"].nunique()
    audit["shared_horizon_count"] = metrics["horizon_minutes"].nunique()
    audit["common_data_contract_passed"] = True
    audit.to_csv(OUTPUT, index=False, float_format="%.3g")
    failed = audit.loc[~audit["passed"]]
    if not failed.empty:
        raise AssertionError("Figure QA failed:\n" + failed[["toolchain", "figure_id", "reason"]].to_string(index=False))
    print(f"Validated {len(audit)} Week-5 images and the common 12-row metric grid")
    print(f"Saved audit: {OUTPUT}")


if __name__ == "__main__":
    main()
