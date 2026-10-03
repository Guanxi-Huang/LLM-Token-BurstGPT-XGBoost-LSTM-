"""Run the complete Week-2 series, baseline, evaluation, and EDA pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    PROJECT_ROOT / "src" / "02_build_series.py",
    PROJECT_ROOT / "src" / "04_baselines.py",
    PROJECT_ROOT / "scripts" / "build_eda_notebook.py",
    PROJECT_ROOT / "scripts" / "validate_week2_outputs.py",
)


def main() -> None:
    for stage in STAGES:
        print(f"\n=== Running {stage.relative_to(PROJECT_ROOT)} ===", flush=True)
        subprocess.run([sys.executable, str(stage)], cwd=PROJECT_ROOT, check=True)
    print("\nWeek-2 pipeline completed successfully.")


if __name__ == "__main__":
    main()
