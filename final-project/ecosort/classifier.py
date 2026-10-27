"""Waste Classifier sub-system (the AI sub-system).

Serves the technique chosen in the investigation — MobileNetV2 transfer
learning at 224x224 — and keeps the SVM on hand-crafted features as a
lightweight, interpretable fallback for hardware that cannot run TensorFlow,
exactly as the report recommends.

Both backends return the same :class:`Prediction`, so every downstream
sub-system (routing, logging, dashboard) is blind to which technique produced
the score. That is what makes the fallback a real option rather than a fork in
the codebase.

Why the TensorFlow probe runs in a subprocess: a mismatched TensorFlow build
can abort the interpreter outright (``libc++abi: terminating due to uncaught
exception``) instead of raising ``ImportError``. A crash cannot be caught in
process, so availability is established once, out of process, and cached.

Robust mode (``predict(..., robust=True)``) averages the network over several
centre crops and their mirror images. It exists because the model is sensitive to
how much of the frame the item fills: measured on a stratified held-out sample,
phone-style framing scores 0.42 with a single pass and 0.70 with six, while clean
studio images go the other way (0.87 down to 0.85). It is therefore opt-in and
used for unposed captures only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import CNN_INPUT_SIZE, MODEL_DIR, SVM_INPUT_SIZE, class_names
from .preprocess import (
    ROBUST_SCALES,
    centre_crop,
    extract_features,
    prepare_for_cnn,
    trim_frame,
)

MODEL_BACKENDS: tuple[str, ...] = ("cnn", "svm")

#: Candidate CNN artefacts, best first. Recipe D ships MobileNetV3; the earlier
#: MobileNetV2 recipe is still loadable so an older set of artefacts keeps working.
CNN_MODEL_FILES: tuple[str, ...] = ("mobilenetv3_224.keras", "mobilenetv2_224.keras")
CNN_MODEL_FILE = CNN_MODEL_FILES[0]
SVM_MODEL_FILE = "svm_hog_glcm.pkl"

_PROBE_TIMEOUT_SECONDS = 180


class BackendUnavailableError(RuntimeError):
    """Raised when the requested classifier backend cannot be loaded."""


class ModelArtifactMissing(FileNotFoundError):
    """Raised when the trained artefact for a backend is not present."""


@dataclass(frozen=True)
class Prediction:
    """One classifier output.

    ``probabilities`` keeps the full six-way distribution rather than only the
    winner, so the UI can show what the model nearly chose and the report can
    analyse confusion.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    backend: str

    @property
    def runner_up(self) -> tuple[str, float] | None:
        """The second most likely class and its probability, if there is one."""
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[1] if len(ranked) > 1 else None

    def as_dict(self) -> dict[str, Any]:
        runner_up = self.runner_up
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
            "runner_up": {"label": runner_up[0], "probability": round(runner_up[1], 4)}
            if runner_up else None,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


def _probe_cache_path() -> Path:
    return MODEL_DIR.parent / ".cache" / "tensorflow_probe.json"


def tensorflow_available(force: bool = False) -> bool:
    """Check, out of process, whether TensorFlow can actually be imported.

    The result is cached per interpreter path, so the app pays the cost at most
    once per environment and never aborts mid-demo.
    """
    cache = _probe_cache_path()
    if not force and cache.is_file():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("executable") == sys.executable:
                return bool(cached.get("available"))
        except (json.JSONDecodeError, OSError):
            pass

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import tensorflow"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        available = result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        available = False

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"executable": sys.executable, "available": available}))
    except OSError:
        pass

    return available


class WasteClassifier:
    """Loads a trained technique and turns a photo into a :class:`Prediction`."""

    def __init__(
        self,
        model_dir: Path | str = MODEL_DIR,
        backend: str = "cnn",
        allow_fallback: bool = True,
        only_member: str | None = None,
    ) -> None:
        if backend not in MODEL_BACKENDS:
            raise ValueError(f"backend must be one of {MODEL_BACKENDS}, got '{backend}'")

        self.model_dir = Path(model_dir)
        self.labels = class_names()
        self.allow_fallback = allow_fallback
        #: When set, the CNN backend loads only this artefact instead of blending
        #: every file it finds. The interface uses it to show one model at a time.
        self.only_member = only_member
        self._requested_backend = backend
        self._resolved_backend: str | None = None
        self._cnn: Any = None
        self._cnn_members: list[tuple[str, Any, float]] = []
        self._temperatures: dict[str, float] = {}
        self._svm: Any = None
        self._load_notes: list[str] = []

    # ------------------------------------------------------------------ setup

    @property
    def backend(self) -> str:
        """The backend that will actually serve predictions."""
        if self._resolved_backend is None:
            self._resolve()
        return self._resolved_backend  # type: ignore[return-value]

    def _resolve(self) -> None:
        """Decide which backend to serve, honouring the fallback policy."""
        notes: list[str] = []
        for candidate in self._backend_order():
            try:
                self._ensure_loaded(candidate)
            except (BackendUnavailableError, ModelArtifactMissing) as exc:
                notes.append(f"{candidate}: {exc}")
                continue
            self._resolved_backend = candidate
            self._load_notes = notes
            return

        detail = "; ".join(notes) if notes else "no models found"
        raise BackendUnavailableError(
            f"No classifier backend could be loaded ({detail}). "
            f"Run the training notebook, or place {' / '.join(CNN_MODEL_FILES)} "
            f"or {SVM_MODEL_FILE} in {self.model_dir}."
        )

    def _backend_order(self) -> list[str]:
        if self._requested_backend == "svm":
            return ["svm", "cnn"] if self.allow_fallback else ["svm"]
        return ["cnn", "svm"] if self.allow_fallback else ["cnn"]

    def _ensure_loaded(self, backend: str) -> None:
        if backend == "cnn":
            self._load_cnn()
        elif backend == "svm":
            self._load_svm()
        else:
            raise ValueError(f"unknown backend '{backend}'")

    def _cnn_path(self) -> Path:
        """The CNN artefact to load, preferring the current recipe's file."""
        for name in CNN_MODEL_FILES:
            candidate = self.model_dir / name
            if candidate.is_file():
                return candidate
        return self.model_dir / CNN_MODEL_FILE

    def _svm_path(self) -> Path:
        return self.model_dir / SVM_MODEL_FILE

    def _load_cnn(self) -> None:
        if self._cnn_members:
            return
        available = [self.model_dir / name for name in CNN_MODEL_FILES
                     if (self.model_dir / name).is_file()
                     and (self.only_member is None or name == self.only_member)]
        if not available:
            raise ModelArtifactMissing(f"missing {CNN_MODEL_FILE} in {self.model_dir}")
        if not tensorflow_available():
            raise BackendUnavailableError(
                "TensorFlow cannot be imported in this environment "
                "(see .cache/tensorflow_probe.json)"
            )

        # Keras 3 is imported directly: tensorflow re-exports it lazily, so the
        # attribute is invisible to a static checker even though it exists.
        import keras  # noqa: PLC0415 — deliberate: only touch Keras when TF works

        weights = self._ensemble_weights([p.name for p in available])
        self._temperatures = self._calibration_map([p.name for p in available])
        members: list[tuple[str, Any, float]] = []
        for path in available:
            try:
                model = keras.models.load_model(str(path), compile=False)
            except (OSError, ValueError) as exc:
                raise BackendUnavailableError(f"could not load {path.name}: {exc}") from exc
            members.append((path.name, model, weights.get(path.name, 0.0)))

        total = sum(weight for _, _, weight in members)
        if total <= 0:                     # no calibrated blend: share equally
            share = 1.0 / len(members)
            members = [(name, m, share) for name, m, _ in members]
        else:
            members = [(name, m, w / total) for name, m, w in members]
        self._cnn_members = members
        self._cnn = members[0][1]

    def _ensemble_weights(self, names: list[str]) -> dict[str, float]:
        """Blend weights for the available CNN artefacts, read from metrics.json.

        Two architectures make different mistakes, so averaging them is worth a
        couple of points of accuracy. The weights are chosen on the validation set
        during training; if the block is missing, callers fall back to equal shares.
        """
        path = self.model_dir / "metrics.json"
        if not path.is_file():
            return {}
        try:
            block = json.loads(path.read_text()).get("ensemble", {}).get("cnn_weights", {})
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return {}
        if not isinstance(block, dict):
            return {}
        found: dict[str, float] = {}
        for name in names:
            value = block.get(name)
            if isinstance(value, (int, float)):
                found[name] = float(value)
        return found

    def _load_svm(self) -> None:
        if self._svm is not None:
            return
        path = self._svm_path()
        if not path.is_file():
            raise ModelArtifactMissing(f"missing {path.name} in {self.model_dir}")

        import joblib  # noqa: PLC0415

        try:
            bundle = joblib.load(path)
        except (OSError, ValueError) as exc:
            raise BackendUnavailableError(f"could not load {path.name}: {exc}") from exc

        if isinstance(bundle, dict) and "pipeline" in bundle:
            self._svm = bundle["pipeline"]
            names = bundle.get("class_names")
            if names:
                self.labels = list(names)
        else:  # a bare fitted pipeline from an older artefact
            self._svm = bundle

    # -------------------------------------------------------------- inference

    def predict(self, image: Image.Image, backend: str | None = None,
                robust: bool = False) -> Prediction:
        """Classify one photo and return the label with its confidence.

        ``robust`` trades latency for accuracy on unposed photos; see
        :meth:`_predict_cnn`.
        """
        chosen = backend or self.backend
        if chosen not in MODEL_BACKENDS:
            raise ValueError(f"backend must be one of {MODEL_BACKENDS}, got '{chosen}'")
        self._ensure_loaded(chosen)

        if chosen == "cnn":
            probabilities = self._predict_cnn(image, robust=robust)
        else:
            probabilities = self._predict_svm(image)

        named = {label: float(probabilities[i]) for i, label in enumerate(self.labels)}
        label = max(named, key=lambda name: named[name])
        return Prediction(
            label=label,
            confidence=named[label],
            probabilities=named,
            backend=chosen,
        )

    def _calibration_map(self, names: list[str]) -> dict[str, float]:
        """Temperature per artefact, read from metrics.json.

        Temperature scaling (Guo et al., 2017) rescales a model's probabilities
        without changing which class wins, so it cannot alter accuracy on its own.
        It matters when two models are blended: an overconfident member otherwise
        dominates the average. Measured on the saved artefacts, MobileNetV2 needed
        T = 1.53 (it was overconfident) and MobileNetV3 T = 0.75 (underconfident).
        """
        path = self.model_dir / "metrics.json"
        if not path.is_file():
            return {}
        try:
            block = json.loads(path.read_text()).get("ensemble", {}).get("temperatures", {})
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return {}
        if not isinstance(block, dict):
            return {}
        found: dict[str, float] = {}
        for name in names:
            value = block.get(name)
            if isinstance(value, (int, float)) and float(value) > 0:
                found[name] = float(value)
        return found

    def _calibrate(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to one probability vector."""
        if temperature == 1.0:
            return probabilities
        logits = np.log(np.clip(probabilities, 1e-7, 1.0)) / temperature
        logits -= logits.max()
        scaled = np.exp(logits)
        total = scaled.sum()
        return scaled / total if total > 0 else probabilities

    def _predict_cnn(self, image: Image.Image, robust: bool = False) -> np.ndarray:
        """Score one photo with the CNN ensemble.

        Every available artefact is run and combined with its calibrated weight.
        MobileNetV2 and MobileNetV3 disagree on roughly one item in ten, and
        blending them lifts held-out accuracy from 0.903 to 0.926.

        With ``robust`` each member is additionally averaged over several centre
        crops and their mirror images. That costs six forwards per member. It
        exists because the models are sensitive to how much of the frame the item
        fills: on phone-style photos multi-scale lifts accuracy from about 0.42 to
        0.70, while on clean studio images it is slightly worse. It is therefore
        off by default and switched on for unposed captures only.
        """
        # A screenshot keeps its window chrome, and a dark border is enough to shift
        # the verdict. Trimming is a no-op on an ordinary photo.
        image = trim_frame(image)

        if not robust:
            batch = prepare_for_cnn(image, CNN_INPUT_SIZE)[None, ...]
            plain = np.zeros(len(self.labels), dtype="float64")
            for name, model, weight in self._cnn_members:
                scored = self._normalise(model.predict(batch, verbose=0)[0])
                plain += weight * self._calibrate(scored,
                                                  self._temperatures.get(name, 1.0))
            return plain

        total = np.zeros(len(self.labels), dtype="float64")
        for name, model, weight in self._cnn_members:
            temperature = self._temperatures.get(name, 1.0)
            for fraction in ROBUST_SCALES:
                batch = prepare_for_cnn(centre_crop(image, fraction),
                                        CNN_INPUT_SIZE)[None, ...]
                flipped = batch[:, :, ::-1, :]
                scored = (self._normalise(model.predict(batch, verbose=0)[0])
                          + self._normalise(model.predict(flipped, verbose=0)[0])) / 2
                total += weight * self._calibrate(scored, temperature)
        return total / len(ROBUST_SCALES)

    def _normalise(self, scores: Any) -> np.ndarray:
        """Scale a score vector to sum to one, defensively."""
        values = np.asarray(scores, dtype="float64")
        total = values.sum()
        if total <= 0:
            return np.full(len(self.labels), 1 / len(self.labels))
        return values / total

    def _predict_svm(self, image: Image.Image) -> np.ndarray:
        features = extract_features(image, SVM_INPUT_SIZE).reshape(1, -1)
        scores = np.asarray(self._svm.predict_proba(features)[0], dtype="float64")
        # The SVM was fitted on the full training label set; align to self.labels.
        classes = getattr(self._svm, "classes_", np.arange(len(scores)))
        aligned = np.zeros(len(self.labels))
        for index, cls in enumerate(classes):
            aligned[int(cls)] = scores[index]
        total = aligned.sum()
        return aligned / total if total > 0 else np.full(len(self.labels), 1 / len(self.labels))

    # ------------------------------------------------------------------ info

    def info(self) -> dict[str, Any]:
        """Describe the loaded models, for the UI and the report appendix."""
        details: dict[str, Any] = {
            "requested_backend": self._requested_backend,
            "active_backend": self.backend,
            "class_names": self.labels,
            "fallback_used": self.backend != self._requested_backend,
            "load_notes": self._load_notes,
            "artifacts": {},
        }
        for name, path in (("cnn", self._cnn_path()), ("svm", self._svm_path())):
            details["artifacts"][name] = {
                "file": path.name,
                "present": path.is_file(),
                "size_mb": round(path.stat().st_size / 1e6, 2) if path.is_file() else None,
            }
        details["tensorflow_available"] = tensorflow_available()
        return details

    def warm_up(self) -> None:
        """Run one dummy inference so the first real item is not slow."""
        dummy = Image.new("RGB", (CNN_INPUT_SIZE, CNN_INPUT_SIZE), (128, 128, 128))
        self.predict(dummy)
