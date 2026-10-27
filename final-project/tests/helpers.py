"""Availability checks for optional test resources.

Deliberately free of third-party imports: the test modules build their own
``pytest.mark.skipif`` markers from these plain booleans, which keeps this
module importable by any tooling that inspects the test tree.
"""

from __future__ import annotations

from pathlib import Path

from ecosort.config import DATA_DIR, MODEL_DIR

MODEL_FILES = ("mobilenetv3_224.keras", "mobilenetv2_224.keras", "svm_hog_glcm.pkl")
SPLIT_FILE = MODEL_DIR / "split_indices.json"


def model_available(model_dir: Path = MODEL_DIR) -> bool:
    """True when at least one trained classifier artefact is present."""
    return any((model_dir / name).is_file() for name in MODEL_FILES)


def split_available() -> bool:
    """True when the training notebook's split indices are present."""
    return SPLIT_FILE.is_file()


def data_available() -> bool:
    """True when the TrashNet image folders are present."""
    return DATA_DIR.is_dir()


def full_system_available() -> bool:
    """True when an end-to-end run is genuinely possible."""
    return model_available() and split_available() and data_available()
