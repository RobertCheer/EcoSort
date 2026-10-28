"""Waste Classifier tests.

Two layers: contract tests that run without artefacts (the :class:`Prediction`
shape, backend validation, error paths), and inference tests that skip until
the trained models are present.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from PIL import Image

from ecosort.classifier import (
    MODEL_BACKENDS,
    BackendUnavailableError,
    ModelArtifactMissing,
    Prediction,
    WasteClassifier,
    tensorflow_available,
)
from ecosort.config import CLASS_NAMES, MODEL_DIR

from .helpers import model_available

requires_models = pytest.mark.skipif(
    not model_available(), reason="trained artefacts missing from models/"
)

# ------------------------------------------------------------------ Prediction


def test_prediction_reports_runner_up():
    prediction = Prediction(
        label="glass",
        confidence=0.55,
        probabilities={"glass": 0.55, "plastic": 0.30, "paper": 0.15},
        backend="cnn",
    )
    runner_up = prediction.runner_up
    assert runner_up is not None
    assert runner_up[0] == "plastic"
    assert runner_up[1] == pytest.approx(0.30)


def test_prediction_with_single_class_has_no_runner_up():
    prediction = Prediction(label="glass", confidence=1.0,
                            probabilities={"glass": 1.0}, backend="svm")
    assert prediction.runner_up is None


def test_prediction_as_dict_is_json_safe():
    prediction = Prediction(
        label="metal",
        confidence=0.81234,
        probabilities={"metal": 0.81234, "glass": 0.18766},
        backend="cnn",
    )
    payload = prediction.as_dict()
    assert payload["label"] == "metal"
    assert payload["confidence"] == 0.8123
    assert payload["runner_up"] == {"label": "glass", "probability": 0.1877}
    assert set(payload["probabilities"]) == {"metal", "glass"}


def test_prediction_is_frozen():
    prediction = Prediction(label="glass", confidence=0.9,
                            probabilities={"glass": 0.9}, backend="cnn")
    with pytest.raises(FrozenInstanceError):
        prediction.label = "paper"  # type: ignore[misc]


# -------------------------------------------------------------------- contract


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        WasteClassifier(backend="resnet")


def test_missing_artifacts_raise_clear_error(tmp_path):
    classifier = WasteClassifier(model_dir=tmp_path, backend="svm", allow_fallback=False)
    with pytest.raises((ModelArtifactMissing, BackendUnavailableError)):
        _ = classifier.backend


def test_fallback_reports_which_backend_served(tmp_path):
    """With no artefacts at all, the error must name both attempts."""
    classifier = WasteClassifier(model_dir=tmp_path, backend="cnn", allow_fallback=True)
    with pytest.raises(BackendUnavailableError) as excinfo:
        classifier.backend
    message = str(excinfo.value)
    assert "cnn" in message and "svm" in message


def test_tensorflow_probe_returns_bool():
    assert isinstance(tensorflow_available(), bool)


def test_model_backends_are_the_documented_pair():
    assert MODEL_BACKENDS == ("cnn", "svm")


# ------------------------------------------------------------------- inference


@requires_models
def test_predict_on_a_real_artifact():
    classifier = WasteClassifier(model_dir=MODEL_DIR, backend="svm", allow_fallback=True)
    image = Image.new("RGB", (224, 224), (90, 140, 70))
    prediction = classifier.predict(image)

    assert prediction.label in CLASS_NAMES
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == set(CLASS_NAMES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    top_label = max(prediction.probabilities.items(), key=lambda item: item[1])[0]
    assert prediction.label == top_label


@requires_models
def test_predictions_are_deterministic():
    classifier = WasteClassifier(model_dir=MODEL_DIR, backend="svm", allow_fallback=True)
    image = Image.new("RGB", (224, 224), (200, 60, 60))
    first = classifier.predict(image)
    second = classifier.predict(image)
    assert first.label == second.label
    assert first.confidence == pytest.approx(second.confidence)


@requires_models
def test_info_reports_artifact_presence():
    classifier = WasteClassifier(model_dir=MODEL_DIR, backend="svm", allow_fallback=True)
    info = classifier.info()
    assert info["active_backend"] in MODEL_BACKENDS
    assert info["class_names"] == CLASS_NAMES
    assert set(info["artifacts"]) == {"cnn", "svm"}
    assert any(detail["present"] for detail in info["artifacts"].values())


@requires_models
def test_warm_up_runs():
    classifier = WasteClassifier(model_dir=MODEL_DIR, backend="svm", allow_fallback=True)
    classifier.warm_up()  # must not raise


@requires_models
def test_probabilities_are_a_valid_distribution_for_uniform_input():
    classifier = WasteClassifier(model_dir=MODEL_DIR, backend="svm", allow_fallback=True)
    prediction = classifier.predict(Image.new("RGB", (224, 224), (128, 128, 128)))
    total = sum(prediction.probabilities.values())
    assert np.isfinite(total)
    assert total == pytest.approx(1.0, abs=1e-6)


@requires_models
def test_robust_mode_still_returns_a_valid_distribution():
    """Six passes are averaged; the output must remain a probability distribution."""
    if not tensorflow_available():
        pytest.skip("TensorFlow is unavailable in this environment")

    classifier = WasteClassifier(backend="cnn")
    image = Image.new("RGB", (320, 240), (120, 130, 140))

    plain = classifier.predict(image)
    robust = classifier.predict(image, robust=True)

    for prediction in (plain, robust):
        assert prediction.backend == "cnn"
        assert prediction.label in CLASS_NAMES
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= prediction.confidence <= 1.0
