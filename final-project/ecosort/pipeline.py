"""Integration: one item, end to end.

This is what turns six sub-systems into a system. Every item follows the same
fixed sequence — capture, pre-process, classify, route, actuate, log — and the
dashboard reads only from the log, so what the operator sees is always a
function of what actually happened.

    image -> Capture -> Pre-process -> Classifier -> Router -> Actuator
                                            |           |
                                            +---> Logger <-- Dashboard

The order matters for the test plan: logging happens after routing, so a
decision can never be displayed without also being recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .actuation import ActuationState, SimulatedActuator
from .capture import CapturedItem, ImageCapture
from .classifier import Prediction, WasteClassifier
from .config import LOG_DIR, MODEL_DIR, theta_for_backend
from .dashboard import Dashboard
from .logging_db import LogEntry, Logger, WasteClass
from .routing import Decision, route


@dataclass
class ItemResult:
    """Everything that happened to one item, in one object."""

    item_id: str
    captured: CapturedItem
    prediction: Prediction
    decision: Decision
    actuation: ActuationState
    log_entry: LogEntry
    latency_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    robust: bool = False

    @property
    def was_correct(self) -> bool | None:
        """Whether the classification matched ground truth, when known."""
        return self.log_entry.is_correct

    def summary(self) -> dict[str, Any]:
        """Flat view for the UI, the tests and the report appendix."""
        return {
            "item_id": self.item_id,
            "predicted_class": self.prediction.label,
            "confidence": round(self.prediction.confidence, 4),
            "backend": self.prediction.backend,
            "routed_to": self.decision.routed_to,
            "destination": self.decision.destination_label,
            "is_review": self.decision.is_review,
            "theta": round(self.decision.theta, 4),
            "true_class": self.captured.true_class,
            "was_correct": self.was_correct,
            "latency_ms": round(self.latency_ms, 2),
            "robust": self.robust,
        }


class EcoSortSystem:
    """The integrated EcoSort station."""

    def __init__(
        self,
        model_dir: Path | str = MODEL_DIR,
        backend: str = "cnn",
        theta: float | None = None,
        log_path: Path | str | None = None,
        allow_fallback: bool = True,
        persist: bool = True,
        only_member: str | None = None,
        classifier: WasteClassifier | None = None,
    ) -> None:
        self.theta = float(theta) if theta is not None else theta_for_backend("cnn")
        self.capture = ImageCapture()
        # A caller can hand in an already-loaded classifier. The interface does this
        # so that changing the threshold does not reload the model files.
        self.classifier = classifier if classifier is not None else WasteClassifier(
            model_dir=model_dir, backend=backend, allow_fallback=allow_fallback,
            only_member=only_member,
        )
        if theta is None:
            # The threshold belongs to the backend that actually serves, which
            # may differ from the one requested if a fallback happened.
            self.theta = theta_for_backend(self.classifier.backend)
        self.log_path = Path(log_path) if log_path else LOG_DIR / "ecosort_decision_log.csv"
        # Reload the log that is already on disk. Without this, a restart showed an
        # empty dashboard and an empty review queue even though every earlier decision
        # was still in the CSV, including the photo reference each queued item needs.
        self.logger = Logger.from_csv(self.log_path)
        self.dashboard = Dashboard(self.logger)
        self.actuator = SimulatedActuator()
        self.persist = persist
        self._counter = 0

    # -------------------------------------------------------------- processing

    def process(
        self,
        image: Image.Image | bytes | Path | str,
        item_id: str | None = None,
        true_class: str | None = None,
        backend: str | None = None,
        robust: bool = False,
        source_path: str | None = None,
    ) -> ItemResult:
        """Run one item through the full pipeline and return everything about it.

        ``robust`` is passed to the classifier; it costs six forward passes and
        buys accuracy on unposed photos (see ``WasteClassifier._predict_cnn``).
        """
        started = time.perf_counter()

        tick = time.perf_counter()
        captured = self._capture(image, item_id, true_class, source_path)
        capture_ms = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        prediction = self.classifier.predict(captured.image, backend=backend, robust=robust)
        classify_ms = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        decision = route(prediction, theta=self.theta)
        route_ms = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        actuation = self.actuator.activate(decision)
        actuation_ms = (time.perf_counter() - tick) * 1000

        entry = LogEntry(
            item_id=captured.item_id,
            predicted_class=WasteClass(prediction.label),
            confidence=float(prediction.confidence),
            routed_to=decision.routed_to,
            is_review=decision.is_review,
            backend=prediction.backend,
            true_class=WasteClass(captured.true_class) if captured.true_class else None,
            image_ref=captured.source_path,
            theta=float(decision.theta),
        )
        second = prediction.runner_up
        if second is not None:
            entry.runner_up = str(second[0])
            entry.runner_up_confidence = float(second[1])

        tick = time.perf_counter()
        self.logger.log_decision(entry)
        if self.persist:
            self.logger.append_csv(self.log_path)
        log_ms = (time.perf_counter() - tick) * 1000

        latency_ms = (time.perf_counter() - started) * 1000
        return ItemResult(
            item_id=captured.item_id,
            captured=captured,
            prediction=prediction,
            decision=decision,
            actuation=actuation,
            log_entry=entry,
            latency_ms=latency_ms,
            robust=robust,
            timings_ms={
                "capture": round(capture_ms, 3),
                "classify": round(classify_ms, 3),
                "route": round(route_ms, 3),
                "actuate": round(actuation_ms, 3),
                "log": round(log_ms, 3),
            },
        )

    def _capture(
        self,
        image: Image.Image | bytes | Path | str,
        item_id: str | None,
        true_class: str | None,
        source_path: str | None = None,
    ) -> CapturedItem:
        if isinstance(image, (bytes, bytearray)):
            item = self.capture.from_upload(
                bytes(image), item_id=item_id, true_class=true_class
            )
            if source_path is not None:
                # The interface hands over bytes for a held-out sample. The path is
                # still known, so keep it: the review queue needs it to show the photo.
                return CapturedItem(
                    item_id=item.item_id, image=item.image, source=item.source,
                    true_class=item.true_class, captured_at=item.captured_at,
                    source_path=source_path,
                )
            return item
        if isinstance(image, (Path, str)):
            return self.capture.from_path(image, item_id=item_id, true_class=true_class)
        if isinstance(image, Image.Image):
            item = self.capture.from_frame(image, item_id=item_id, true_class=true_class)
            if source_path is not None:
                # Rebuild with the path so the log can point back at the photo.
                return CapturedItem(
                    item_id=item.item_id, image=item.image, source=item.source,
                    true_class=item.true_class, captured_at=item.captured_at,
                    source_path=source_path,
                )
            return item
        raise TypeError(f"Unsupported image input: {type(image).__name__}")

    def process_batch(
        self,
        items: list[CapturedItem] | None = None,
        count: int = 20,
        split: str = "test",
        backend: str | None = None,
        progress: Any = None,
    ) -> list[ItemResult]:
        """Run a batch of items, by default held-out samples.

        Args:
            items: explicit items to process. When omitted, ``count`` held-out
                samples are drawn from ``split``.
            count: how many samples to draw when ``items`` is omitted.
            split: which split to sample from.
            backend: classifier backend override.
            progress: optional callable ``(done, total)`` for a progress bar.
        """
        batch = items if items is not None else self.capture.sample_items(count=count, split=split)
        results: list[ItemResult] = []
        for done, item in enumerate(batch, start=1):
            results.append(
                self.process(
                    item.image,
                    item_id=item.item_id,
                    true_class=item.true_class,
                    backend=backend,
                    source_path=item.source_path,
                )
            )
            if progress is not None:
                progress(done, len(batch))
        return results

    # -------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Dashboard totals plus a little system context."""
        payload = self.dashboard.summary()
        payload.update({
            "theta": round(self.theta, 4),
            "active_backend": self.classifier.backend,
            "actuations": len(self.actuator),
        })
        return payload

    def save_log(self, path: Path | str | None = None) -> Path:
        """Write the full log to CSV and return where it went."""
        target = Path(path) if path else self.log_path
        self.logger.to_csv(target)
        return target

    def reset(self) -> None:
        """Clear the log, the actuator and the dashboard for a fresh demo.

        The CSV is removed as well. Clearing memory alone left every old row on disk,
        so the next append rewrote them and the log doubled.
        """
        self.logger.clear()
        self.actuator.clear_history()
        self._counter = 0
        try:
            self.log_path.unlink(missing_ok=True)
        except OSError as exc:
            # A read-only or locked file must not break the reset; the in-memory
            # log is already clear, so the station stays usable.
            print(f"warning: could not remove {self.log_path}: {exc}")
