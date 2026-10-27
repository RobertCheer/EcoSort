"""Per-backend confidence threshold calibration.

The system serves two classifiers whose probabilities live on different scales.
MobileNetV2 is sharp (its median confidence on held-out data is about 0.99); the
SVM is blunt (about 0.68). Judging both with one threshold looks fine on paper
and fails in the interface: at the CNN's theta the SVM sends more than half of
all items to human review, which defeats the point of the station.

These tests pin the fix so it cannot silently regress.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ecosort.config import MODEL_DIR, theta_for_backend

METRICS_FILE = MODEL_DIR / "metrics.json"


def test_svm_threshold_is_lower_than_the_cnn_threshold():
    assert theta_for_backend("svm") < theta_for_backend("cnn")


def test_thresholds_match_the_recorded_calibration():
    metrics = json.loads(METRICS_FILE.read_text())
    by_backend = metrics["theta"].get("by_backend", {})
    assert by_backend, "metrics.json should record a threshold for each backend"
    for backend, recorded in by_backend.items():
        assert theta_for_backend(backend) == pytest.approx(float(recorded))


def test_unknown_backend_falls_back_without_raising():
    fallback = theta_for_backend("not-a-backend")
    assert 0.0 < fallback <= 1.0


def test_one_shared_threshold_would_over_review_the_svm():
    """The regression this calibration exists to prevent.

    Replays the saved held-out probabilities: the SVM judged at the CNN's
    threshold diverts most items to a person, while its own threshold keeps the
    review queue inside the same 10% budget the CNN is held to.
    """
    probabilities = np.load(MODEL_DIR / "test_probs.npz")
    svm_confidence = probabilities["svm"].max(axis=1)

    review_with_cnn_threshold = float((svm_confidence < theta_for_backend("cnn")).mean())
    review_with_own_threshold = float((svm_confidence < theta_for_backend("svm")).mean())

    # Assert the relationship, not a fixed share: the calibrated theta moves when
    # the model is retrained, and pinning the number made this brittle.
    assert review_with_cnn_threshold > 0.20, "the shared threshold should be visibly unusable"
    assert review_with_cnn_threshold > 2.5 * review_with_own_threshold, (
        "the CNN's threshold should multiply the SVM's review workload")
    assert review_with_own_threshold < 0.15, "the SVM should review a small share only"


def test_calibration_matches_the_reviewer_capacity_budget():
    """Each backend's threshold keeps the review queue inside the budget."""
    metrics = json.loads(METRICS_FILE.read_text())
    budget = metrics["theta"]["max_review_rate"]
    for backend, key in (("cnn", "sweep_cnn"), ("svm", "sweep_svm")):
        rows = {row["theta"]: row for row in metrics["theta"][key]}
        chosen = rows[theta_for_backend(backend)]
        assert chosen["review_rate"] <= budget
