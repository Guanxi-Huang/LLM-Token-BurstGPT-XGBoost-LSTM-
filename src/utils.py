"""Common paths and reproducibility helpers for the research workflow."""

from __future__ import annotations

import os
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42


def project_path(*parts: str) -> Path:
    """Return an absolute path rooted at the project directory."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_output_directories() -> None:
    """Create only derived-data directories; never touch raw data in Dataset/."""
    for relative_path in (
        "data/processed",
        "models",
        "outputs/figures",
        "outputs/tables",
    ):
        project_path(relative_path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set available pseudo-random generators for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
