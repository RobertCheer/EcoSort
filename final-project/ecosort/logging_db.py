"""Logging sub-system.

Records one entry per item the system processes. This is the single source of
truth the dashboard aggregates over — the dashboard keeps no counters of its
own, so the test-plan requirement "dashboard totals equal log totals" holds by
construction rather than by careful bookkeeping.

Adapted from the assignment's logging module (owner: Waitun, evaluation and
dashboard lead), extended for the integrated system with thread-safe appends,
CSV round-tripping and ground-truth support for evaluation runs.
"""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class WasteClass(str, Enum):
    """The six recycling categories from REQ-01."""

    GLASS = "glass"
    PAPER = "paper"
    CARDBOARD = "cardboard"
    PLASTIC = "plastic"
    METAL = "metal"
    TRASH = "trash"


@dataclass
class LogEntry:
    """One record per item processed by the EcoSort pipeline.

    Fields are filled in by whichever sub-system produces them:

    * ``predicted_class``, ``confidence`` — the waste classifier
    * ``routed_to``, ``is_review`` — decision and bin-routing
    * ``true_class`` — only available on evaluation runs against the held-out
      set; it is ``None`` in live deployment, where no ground truth exists.
    """

    item_id: str
    predicted_class: WasteClass
    confidence: float
    routed_to: str
    is_review: bool
    backend: str = "cnn"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    true_class: WasteClass | None = None
    #: Set to "human" when an operator resolved the item from the review queue.
    #: A row with is_review and no resolved_by is still waiting for a person.
    resolved_by: str | None = None
    #: The bin the person sent the item to. The model's own routing never opened a
    #: bin for a review item, so this is the only destination the item actually got.
    human_destination: str | None = None
    #: The threshold in force when this item was decided. The review text must quote
    #: this value, not the slider's current value, or the sentence contradicts itself
    #: when the slider moves after the decision.
    theta: float | None = None
    #: The model's second choice, kept so the review queue can show a reviewer what
    #: the near-miss was and offer the right inspection advice.
    #: Path to the source photo, when there is one. Lets the review queue show the
    #: item after a restart instead of "no preview kept".
    image_ref: str | None = None
    runner_up: str | None = None
    runner_up_confidence: float | None = None

    @property
    def is_pending_review(self) -> bool:
        """True when the item went to review and no person has decided it yet."""
        return self.is_review and self.resolved_by is None

    @property
    def is_correct(self) -> bool | None:
        """Whether the prediction matched ground truth, when it is known."""
        if self.true_class is None:
            return None
        return self.predicted_class == self.true_class

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["predicted_class"] = self.predicted_class.value
        record["true_class"] = self.true_class.value if self.true_class else None
        record["is_correct"] = self.is_correct
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> LogEntry:
        raw_true = record.get("true_class")
        return cls(
            item_id=str(record["item_id"]),
            predicted_class=WasteClass(record["predicted_class"]),
            confidence=float(record["confidence"]),
            routed_to=str(record["routed_to"]),
            is_review=_as_bool(record["is_review"]),
            backend=str(record.get("backend") or "cnn"),
            timestamp=str(record.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            true_class=WasteClass(raw_true) if raw_true else None,
            resolved_by=(str(record["resolved_by"]) if record.get("resolved_by") else None),
            human_destination=(str(record["human_destination"])
                               if record.get("human_destination") else None),
            image_ref=(str(record["image_ref"]) if record.get("image_ref") else None),
            theta=(float(record["theta"]) if record.get("theta") else None),
            runner_up=(str(record["runner_up"]) if record.get("runner_up") else None),
            runner_up_confidence=(float(record["runner_up_confidence"])
                                  if record.get("runner_up_confidence") else None),
        )


def _as_bool(value: Any) -> bool:
    """Parse a boolean that may have round-tripped through CSV as text."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


FIELDNAMES = [
    "item_id",
    "predicted_class",
    "confidence",
    "routed_to",
    "is_review",
    "backend",
    "timestamp",
    "true_class",
    "resolved_by",
    "human_destination",
    "runner_up",
    "runner_up_confidence",
    "image_ref",
    "theta",
]


def _write_rows(target: Path, rows: list[LogEntry], append: bool) -> None:
    """Write log rows to CSV, adding the header only when the file is new."""
    target.parent.mkdir(parents=True, exist_ok=True)
    is_empty = not target.is_file() or target.stat().st_size == 0
    mode = "a" if append else "w"
    with target.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if is_empty or not append:
            writer.writeheader()
        for entry in rows:
            writer.writerow(entry.to_dict())


class Logger:
    """Append-only record of every decision the system has made."""

    def __init__(self, entries: list[LogEntry] | None = None) -> None:
        self._entries: list[LogEntry] = list(entries or [])
        # How many entries are already on disk. Without this, append_csv would
        # rewrite the whole log on every call, so a live run would write
        # quadratic numbers of rows and duplicate every earlier entry.
        self._persisted_count = len(self._entries)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self.entries)

    def log_decision(self, entry: LogEntry) -> None:
        """Record one decision."""
        with self._lock:
            self._entries.append(entry)

    @property
    def entries(self) -> list[LogEntry]:
        """Defensive copy, so callers cannot mutate the log in place."""
        with self._lock:
            return list(self._entries)

    def resolve_review(self, item_id: str, true_class: WasteClass,
                       reviewer: str = "human") -> LogEntry:
        """Record a person's decision for one queued item.

        The item keeps its original prediction and confidence. Only the ground
        truth and the resolver change, so the dashboard can separate what the
        model got wrong from what a person put right.
        """
        with self._lock:
            for entry in self._entries:
                if entry.item_id == item_id:
                    if not entry.is_review:
                        raise ValueError(
                            f"{item_id} was routed automatically; only review items "
                            "need a person")
                    # A review item never opened a bin. The person's decision is what
                    # finally sends it somewhere, so map the class to its bin here.
                    from .routing import BIN_MAP  # noqa: PLC0415 - avoids a cycle
                    entry.true_class = true_class
                    entry.resolved_by = reviewer
                    entry.human_destination = BIN_MAP[true_class.value]
                    return entry
            raise KeyError(f"no log entry with item_id {item_id!r}")

    def pending_reviews(self) -> list[LogEntry]:
        """Review items that are still waiting for a person, oldest first."""
        with self._lock:
            return [e for e in self._entries if e.is_pending_review]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._persisted_count = 0

    # ------------------------------------------------------------ persistence

    def to_csv(self, path: Path | str) -> None:
        """Write the whole log to CSV, replacing any existing file."""
        with self._lock:
            rows = list(self._entries)
            if not rows:
                raise ValueError("No entries to write.")
            _write_rows(Path(path), rows, append=False)
            self._persisted_count = len(rows)

    def append_csv(self, path: Path | str) -> None:
        """Append only the entries not yet written, creating the header if needed."""
        with self._lock:
            pending = self._entries[self._persisted_count:]
            if not pending:
                return
            _write_rows(Path(path), pending, append=True)
            self._persisted_count = len(self._entries)

    def to_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([e.to_dict() for e in self.entries], indent=2))

    @classmethod
    def from_csv(cls, path: Path | str) -> Logger:
        """Load a log written by :meth:`to_csv` or :meth:`append_csv`."""
        source = Path(path)
        if not source.is_file():
            return cls()
        with source.open(newline="") as handle:
            return cls([LogEntry.from_dict(row) for row in csv.DictReader(handle)])

    def to_dataframe(self):
        """Expose the log as a pandas DataFrame for the dashboard charts."""
        import pandas as pd  # noqa: PLC0415 — keep the module importable without pandas

        return pd.DataFrame([e.to_dict() for e in self.entries])
