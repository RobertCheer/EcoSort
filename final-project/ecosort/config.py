"""Shared settings for the EcoSort system.

One place for the values that the whole pipeline agrees on, so the classifier,
router, logger and UI cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT.parent / "work" / "dataset-resized"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# The six recycling categories, in the label order both models were trained on.
# This order is fixed: it is the column order of every probability vector.
CLASS_NAMES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CNN_INPUT_SIZE = 224
SVM_INPUT_SIZE = 128

# Confidence threshold for bin routing. Overridden at import time by the value
# calibrated on the validation set during training (models/metrics.json).
THETA = 0.50

#: Design placeholder used before the calibration exists. The sweep picks the
#: most accurate automatic path that fits inside a 10% reviewer-capacity budget;
#: on the current run that lands at 0.50.
_DEFAULT_THETA = 0.50


def _load_trained_theta() -> float:
    """Read the calibrated threshold from the training artefacts, if present.

    Falls back to the design placeholder so the system still runs before the
    sweep has been executed.
    """
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.is_file():
        return _DEFAULT_THETA
    try:
        metrics = json.loads(metrics_path.read_text())
        return float(metrics["theta"]["chosen"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _DEFAULT_THETA


THETA = _load_trained_theta()

#: Fallback thresholds used when ``metrics.json`` does not carry a calibrated
#: value for a backend. The SVM's probabilities are blunt (its median confidence
#: on held-out data is about 0.68 against MobileNetV2's 0.99), so judging both
#: with one threshold would send most SVM items to a person.
_DEFAULT_THETA_BY_BACKEND: dict[str, float] = {"cnn": _DEFAULT_THETA, "svm": 0.35}


def _as_float(value: object) -> float | None:
    """Best-effort float conversion for a value that came out of JSON."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def theta_for_backend(backend: str) -> float:
    """Calibrated confidence threshold for one classifier backend.

    Each backend has its own confidence scale, so each needs its own theta.
    The values are read from ``metrics.json`` under ``theta.by_backend``; when
    that block is missing the CNN falls back to the recorded ``theta.chosen``
    and the SVM to its own default.
    """
    metrics_path = MODEL_DIR / "metrics.json"
    theta_block: dict = {}
    if metrics_path.is_file():
        try:
            theta_block = json.loads(metrics_path.read_text()).get("theta", {})
        except (json.JSONDecodeError, TypeError, ValueError):
            theta_block = {}

    if isinstance(theta_block, dict):
        by_backend = theta_block.get("by_backend") or {}
        if isinstance(by_backend, dict):
            configured = _as_float(by_backend.get(backend))
            if configured is not None:
                return configured
        if backend == "cnn":
            chosen = _as_float(theta_block.get("chosen"))
            if chosen is not None:
                return chosen

    return _DEFAULT_THETA_BY_BACKEND.get(backend, _DEFAULT_THETA)


def class_names() -> list[str]:
    """Label order actually used by the trained artefacts.

    Falls back to the built-in order when the artefact file is missing.
    """
    path = MODEL_DIR / "class_names.json"
    if not path.is_file():
        return list(CLASS_NAMES)
    try:
        names = json.loads(path.read_text()).get("class_names")
    except json.JSONDecodeError:
        return list(CLASS_NAMES)
    return list(names) if names else list(CLASS_NAMES)
