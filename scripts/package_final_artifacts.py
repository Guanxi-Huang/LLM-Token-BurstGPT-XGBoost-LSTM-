"""Freeze compact, shareable tables and figures with a checksum manifest.

Raw traces, processed request-level data, model binaries and row-level predictions
are intentionally excluded.  The resulting ``artifacts/final`` directory can be
versioned and is sufficient to trace every paper claim back to a generated CSV or
PNG.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_SOURCE = ROOT / "outputs" / "tables"
FIGURE_SOURCE = ROOT / "outputs" / "figures"
FINAL_ROOT = ROOT / "artifacts" / "final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_sources() -> list[Path]:
    tables = sorted(TABLE_SOURCE.glob("table_*.csv"))
    tables += sorted(TABLE_SOURCE.glob("xgb_feature_importance_h*.csv"))
    tables += sorted(TABLE_SOURCE.glob("xgb_shap_importance_h*.csv"))
    tables += [
        TABLE_SOURCE / "xgb_tuning_log.csv",
        TABLE_SOURCE / "lstm_tuning_log.csv",
        TABLE_SOURCE / "lstm_history_h15.csv",
    ]

    figures = sorted(FIGURE_SOURCE.glob("fig_*.png"))
    figures += sorted((FIGURE_SOURCE / "w5_python").glob("*.png"))
    figures += sorted((FIGURE_SOURCE / "w5_rstudio").glob("*.png"))
    return [path for path in tables + figures if path.exists()]


def destination_for(source: Path) -> Path:
    if source.suffix.lower() == ".csv":
        return FINAL_ROOT / "tables" / source.name
    if source.parent.name in {"w5_python", "w5_rstudio"}:
        return FINAL_ROOT / "figures" / source.parent.name / source.name
    return FINAL_ROOT / "figures" / source.name


def main() -> None:
    sources = selected_sources()
    if not sources:
        raise FileNotFoundError("No generated tables or figures were found; run the README pipeline first.")

    manifest_rows: list[dict[str, str | int]] = []
    for source in sources:
        destination = destination_for(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest_rows.append(
            {
                "artifact_path": destination.relative_to(ROOT).as_posix(),
                "source_path": source.relative_to(ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest_path = FINAL_ROOT / "manifest_sha256.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact_path", "source_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Packaged {sum(p.suffix.lower() == '.csv' for p in sources)} tables and "
        f"{sum(p.suffix.lower() == '.png' for p in sources)} figures into {FINAL_ROOT}"
    )
    print(f"Manifest: {manifest_path} ({len(manifest_rows)} entries)")


if __name__ == "__main__":
    main()
