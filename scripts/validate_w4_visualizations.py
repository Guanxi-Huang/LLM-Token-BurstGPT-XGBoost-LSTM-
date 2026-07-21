"""Validate the Week-4 Python and RStudio visualization bundles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "outputs" / "tables" / "w4_visualization"
PYTHON = ROOT / "outputs" / "figures" / "w4_python"
RSTUDIO = ROOT / "outputs" / "figures" / "w4_rstudio"


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty visualization artifact: {path}")
    return path


def validate_image(path: Path) -> None:
    require(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        assert width >= 1200 and height >= 700, (path, image.size)
        rgb = image.convert("RGB").resize((160, 100))
        variance = sum(ImageStat.Stat(rgb).var)
        assert variance > 20, f"Image appears blank or nearly blank: {path}"


def main() -> None:
    shared = [
        "w4_sequence_audit.csv", "w4_manual_sequence_check.csv", "w4_features.csv",
        "w4_feature_counts.csv", "w4_training_history.csv", "w4_selected_lstm.csv",
        "w4_model_metrics.csv", "w4_h15_representative_day.csv", "w4_h15_test_bias.csv",
        "python_environment.json", "r_session_info.txt",
    ]
    for name in shared:
        require(VIZ / name)

    py_manifest = pd.read_csv(require(VIZ / "w4_python_figure_manifest.csv"))
    r_manifest = pd.read_csv(require(VIZ / "w4_r_figure_manifest.csv"))
    expected_numbers = [f"W4-{index:02d}" for index in range(1, 10)]
    assert py_manifest["figure_number"].tolist() == expected_numbers
    assert r_manifest["figure_number"].tolist() == expected_numbers
    assert len(py_manifest) == 9 and len(r_manifest) == 9
    for name in py_manifest["python_file"]:
        validate_image(PYTHON / str(name))
    for name in r_manifest["rstudio_file"]:
        validate_image(RSTUDIO / str(name))
    validate_image(PYTHON / "w4_python_overview.png")
    validate_image(RSTUDIO / "w4_rstudio_overview.png")

    metrics = pd.read_csv(VIZ / "w4_model_metrics.csv")
    assert len(metrics.loc[metrics["split"].eq("valid")]) == 12
    assert len(metrics.loc[metrics["split"].eq("test")]) == 12
    assert set(metrics["model_label"]) == {"Persistence", "Seasonal Naive", "XGBoost", "LSTM"}
    assert set(metrics["horizon_minutes"].astype(int)) == {5, 15, 60}
    trace = pd.read_csv(VIZ / "w4_h15_representative_day.csv")
    assert len(trace) == 288
    assert trace["relative_hour"].is_monotonic_increasing
    print("W4 visualization validation passed: 9 Python figures, 9 RStudio figures, 2 overviews, and shared plotting data are complete.")


if __name__ == "__main__":
    main()
