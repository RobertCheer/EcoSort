"""Dashboard sub-system.

Pure aggregation over the logger's entries. The dashboard keeps no counters of
its own, so "dashboard totals equal log totals" is true by construction — there
is no second tally that could drift.

Two live indicators are exposed, and they answer different questions:

* **low-confidence rate** — share of items sent to a person. Needs no ground
  truth, so it is the only honest live signal in deployment.
* **contamination rate** — share of *evaluated* items that were misclassified.
  Requires ``true_class``, so it is available on held-out-set runs and on live
  runs where an operator audits a sample afterwards.
"""

from __future__ import annotations

from typing import Any

from .logging_db import LogEntry, Logger, WasteClass
from .routing import HUMAN_REVIEW

#: Bin order used for stable chart series.
BIN_ORDER = [
    "bin_cardboard",
    "bin_glass",
    "bin_metal",
    "bin_paper",
    "bin_plastic",
    "bin_trash",
    HUMAN_REVIEW,
]


class Dashboard:
    """Running contamination statistics for the operator."""

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    # ------------------------------------------------------------- raw counts

    def total_logged(self) -> int:
        return len(self.logger)

    def per_class_counts(self) -> dict[str, int]:
        """How many items the classifier put in each category."""
        counts = {cls.value: 0 for cls in WasteClass}
        for entry in self.logger.entries:
            counts[entry.predicted_class.value] += 1
        return counts

    def per_bin_counts(self) -> dict[str, int]:
        """How many items went to each physical destination, review included."""
        counts = dict.fromkeys(BIN_ORDER, 0)
        for entry in self.logger.entries:
            counts[entry.routed_to] = counts.get(entry.routed_to, 0) + 1
        return counts

    def review_count(self) -> int:
        return sum(1 for entry in self.logger.entries if entry.is_review)

    def routed_count(self) -> int:
        return self.total_logged() - self.review_count()

    # ------------------------------------------------------------------ rates

    def low_confidence_rate(self) -> float:
        """Share of items diverted to human review — the live workload signal."""
        entries = self.logger.entries
        if not entries:
            return 0.0
        return self.review_count() / len(entries)

    def contamination_rate(self) -> float | None:
        """Share of evaluated items that reached the wrong destination.

        Returns ``None`` when no item carries ground truth, because the number
        would be fabricated rather than measured.
        """
        evaluated = [e for e in self.logger.entries if e.true_class is not None]
        if not evaluated:
            return None
        wrong = sum(1 for e in evaluated if not e.is_correct)
        return wrong / len(evaluated)

    def accuracy_on_evaluated(self) -> float | None:
        """Share of evaluated items the classifier labelled correctly."""
        evaluated = [e for e in self.logger.entries if e.true_class is not None]
        if not evaluated:
            return None
        correct = sum(1 for e in evaluated if e.is_correct)
        return correct / len(evaluated)

    def auto_routed_accuracy(self) -> float | None:
        """Accuracy on the items the system was confident enough to bin itself.

        This is the number the threshold is really protecting: it excludes the
        items handed to a human, so it says how reliable the automatic path is.
        """
        auto = [e for e in self.logger.entries if not e.is_review and e.true_class is not None]
        if not auto:
            return None
        return sum(1 for e in auto if e.is_correct) / len(auto)

    def per_class_accuracy(self) -> dict[str, dict[str, float | int]]:
        """Per-class hits and misses, which expose the weak minority class."""
        stats: dict[str, dict[str, float | int]] = {}
        for cls in WasteClass:
            truth = [e for e in self.logger.entries if e.true_class == cls]
            stats[cls.value] = {
                "evaluated": len(truth),
                "correct": sum(1 for e in truth if e.is_correct),
                "accuracy": (
                    sum(1 for e in truth if e.is_correct) / len(truth) if truth else 0.0
                ),
            }
        return stats

    # -------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        """Everything the operator view needs, in one call."""
        return {
            "total_logged": self.total_logged(),
            "per_class_counts": self.per_class_counts(),
            "per_bin_counts": self.per_bin_counts(),
            "review_count": self.review_count(),
            "routed_count": self.routed_count(),
            "low_confidence_rate": round(self.low_confidence_rate(), 4),
            "contamination_rate": _round_or_none(self.contamination_rate()),
            "accuracy_on_evaluated": _round_or_none(self.accuracy_on_evaluated()),
            "auto_routed_accuracy": _round_or_none(self.auto_routed_accuracy()),
            "per_class_accuracy": self.per_class_accuracy(),
        }

    def recent(self, limit: int = 10) -> list[LogEntry]:
        """The most recent decisions, newest first."""
        return list(reversed(self.logger.entries[-limit:]))

    def timeseries(self):
        """Cumulative accuracy and review rate per item, for the trend chart."""
        import pandas as pd  # noqa: PLC0415

        entries = self.logger.entries
        if not entries:
            return pd.DataFrame(
                [], columns=pd.Index(["index", "timestamp", "cumulative_accuracy", "review_rate"])
            )

        rows: list[dict[str, Any]] = []
        correct = 0
        evaluated = 0
        reviewed = 0
        for index, entry in enumerate(entries, start=1):
            if entry.is_correct is not None:
                evaluated += 1
                correct += int(bool(entry.is_correct))
            if entry.is_review:
                reviewed += 1
            rows.append({
                "index": index,
                "timestamp": entry.timestamp,
                "cumulative_accuracy": correct / evaluated if evaluated else None,
                "review_rate": reviewed / index,
            })
        return pd.DataFrame(rows)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)
