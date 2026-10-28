"""Logging and dashboard tests.

The head-line requirement from the test plan is that dashboard totals equal log
totals exactly. These tests hold that invariant under the cases that would
normally break a hand-maintained counter: an empty log, a partially evaluated
log, a CSV round-trip, and repeated appends.
"""

from __future__ import annotations

import csv

import pytest

from ecosort.dashboard import BIN_ORDER, Dashboard
from ecosort.logging_db import FIELDNAMES, LogEntry, Logger, WasteClass
from ecosort.routing import HUMAN_REVIEW


def make_entry(index: int, predicted: str, confidence: float, review: bool,
               true_class: str | None = None, routed_to: str | None = None) -> LogEntry:
    return LogEntry(
        item_id=f"item_{index:04d}",
        predicted_class=WasteClass(predicted),
        confidence=confidence,
        routed_to=routed_to or (HUMAN_REVIEW if review else f"bin_{predicted}"),
        is_review=review,
        true_class=WasteClass(true_class) if true_class else None,
    )


@pytest.fixture()
def populated() -> tuple[Logger, Dashboard]:
    logger = Logger()
    rows = [
        ("glass", 0.91, False, "glass"),
        ("paper", 0.45, True, "paper"),
        ("plastic", 0.88, False, "glass"),     # confidently wrong
        ("trash", 0.61, False, None),
        ("metal", 0.71, False, "metal"),
    ]
    for index, (predicted, confidence, review, truth) in enumerate(rows):
        logger.log_decision(make_entry(index, predicted, confidence, review, truth))
    return logger, Dashboard(logger)


def test_dashboard_totals_equal_log_totals(populated):
    logger, dashboard = populated
    counts = dashboard.per_class_counts()
    assert sum(counts.values()) == len(logger)
    assert dashboard.total_logged() == len(logger)
    assert sum(dashboard.per_bin_counts().values()) == len(logger)


def test_empty_log_is_handled_without_error():
    dashboard = Dashboard(Logger())
    assert dashboard.total_logged() == 0
    assert dashboard.low_confidence_rate() == 0.0
    assert dashboard.contamination_rate() is None
    assert dashboard.accuracy_on_evaluated() is None
    assert dashboard.auto_routed_accuracy() is None
    assert sum(dashboard.per_class_counts().values()) == 0


def test_low_confidence_rate_counts_reviews(populated):
    _, dashboard = populated
    assert dashboard.review_count() == 1
    assert dashboard.low_confidence_rate() == pytest.approx(1 / 5)


def test_contamination_rate_uses_only_evaluated_items(populated):
    """One confident error out of four items that carry ground truth."""
    _, dashboard = populated
    assert dashboard.contamination_rate() == pytest.approx(0.25)
    assert dashboard.accuracy_on_evaluated() == pytest.approx(0.75)


def test_auto_routed_accuracy_excludes_reviewed_items(populated):
    """Of the three auto-routed items with ground truth, two were right.

    The reviewed paper item is correct but must not count here: this number
    measures how reliable the automatic path alone is.
    """
    _, dashboard = populated
    assert dashboard.auto_routed_accuracy() == pytest.approx(2 / 3)


def test_per_class_accuracy_exposes_the_weak_class(populated):
    """Two glass items were evaluated and only one was classified correctly."""
    _, dashboard = populated
    stats = dashboard.per_class_accuracy()
    assert stats["glass"]["evaluated"] == 2
    assert stats["glass"]["correct"] == 1
    assert stats["glass"]["accuracy"] == pytest.approx(0.5)
    assert stats["metal"]["accuracy"] == pytest.approx(1.0)
    assert stats["cardboard"]["evaluated"] == 0


def test_contamination_is_none_without_ground_truth():
    logger = Logger()
    logger.log_decision(make_entry(0, "glass", 0.9, False, None))
    dashboard = Dashboard(logger)
    assert dashboard.contamination_rate() is None
    # Without ground truth the live view still reports its own workload signal.
    assert dashboard.low_confidence_rate() == 0.0
    assert dashboard.total_logged() == 1


def test_summary_keys_and_shapes(populated):
    _, dashboard = populated
    summary = dashboard.summary()
    assert summary["total_logged"] == 5
    assert set(summary["per_class_counts"]) == {c.value for c in WasteClass}
    assert set(summary["per_bin_counts"]) == set(BIN_ORDER)
    assert 0.0 <= summary["low_confidence_rate"] <= 1.0


def test_csv_round_trip_preserves_every_field(tmp_path):
    logger = Logger()
    logger.log_decision(make_entry(0, "glass", 0.91, False, "glass"))
    logger.log_decision(make_entry(1, "paper", 0.40, True, "trash"))
    path = tmp_path / "log.csv"
    logger.to_csv(path)

    reloaded = Logger.from_csv(path)
    assert len(reloaded) == 2
    first, second = reloaded.entries
    assert first.predicted_class is WasteClass.GLASS
    assert first.is_review is False
    assert second.is_review is True
    assert second.true_class is WasteClass.TRASH
    assert first.timestamp  # survived the round trip


def test_csv_header_matches_the_documented_schema(tmp_path):
    logger = Logger()
    logger.log_decision(make_entry(0, "glass", 0.9, False, "glass"))
    path = tmp_path / "log.csv"
    logger.to_csv(path)
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == FIELDNAMES


def test_append_csv_writes_only_new_entries(tmp_path):
    """Repeated appends must not re-write old rows or duplicate the header."""
    path = tmp_path / "log.csv"
    logger = Logger()
    logger.log_decision(make_entry(0, "glass", 0.9, False))
    logger.append_csv(path)
    logger.log_decision(make_entry(1, "paper", 0.8, False))
    logger.append_csv(path)
    logger.append_csv(path)          # nothing new: must be a no-op

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == FIELDNAMES               # header written exactly once
    assert len(rows) == 3                      # header + two entries
    assert rows[1][0] == "item_0000"
    assert rows[2][0] == "item_0001"


def test_empty_logger_refuses_to_write_an_empty_file(tmp_path):
    with pytest.raises(ValueError):
        Logger().to_csv(tmp_path / "empty.csv")


def test_entries_property_is_a_defensive_copy():
    logger = Logger()
    logger.log_decision(make_entry(0, "glass", 0.9, False))
    snapshot = logger.entries
    snapshot.clear()
    assert len(logger) == 1


def test_recent_returns_newest_first(populated):
    _, dashboard = populated
    recent = dashboard.recent(limit=2)
    assert [e.item_id for e in recent] == ["item_0004", "item_0003"]


def test_timeseries_is_monotonic_in_index(populated):
    _, dashboard = populated
    series = dashboard.timeseries()
    assert list(series["index"]) == [1, 2, 3, 4, 5]
    assert series["cumulative_accuracy"].iloc[-1] == pytest.approx(0.75)


def test_clear_empties_the_log(populated):
    logger, dashboard = populated
    logger.clear()
    assert len(logger) == 0
    assert dashboard.total_logged() == 0
