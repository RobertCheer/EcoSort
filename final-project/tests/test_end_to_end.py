"""End-to-end tests: the integration and system rows of the test plan.

These exercise the requirement that actually defines the product — that a batch
of held-out items each reaches the right bin (or the review queue), gets logged,
appears correctly on the dashboard, and stays inside the one-second budget.

They are skipped until a trained model, the split indices and the dataset are
all present, because an end-to-end test without a model would be theatre.
"""

from __future__ import annotations

import pytest

from ecosort.config import THETA
from ecosort.logging_db import WasteClass
from ecosort.pipeline import EcoSortSystem
from ecosort.routing import HUMAN_REVIEW

from .helpers import full_system_available

requires_full_system = pytest.mark.skipif(
    not full_system_available(),
    reason="end-to-end tests need the dataset, the split indices and a trained model",
)


@pytest.fixture()
def system(tmp_path) -> EcoSortSystem:
    """A system with logging isolated to this test's temp directory."""
    return EcoSortSystem(
        backend="svm",
        theta=THETA,
        log_path=tmp_path / "log.csv",
        allow_fallback=True,
    )


@requires_full_system
def test_every_item_is_logged_and_appears_on_the_dashboard(system):
    results = system.process_batch(count=12, split="test")

    assert len(results) == 12
    assert len(system.logger) == 12

    summary = system.dashboard.summary()
    assert summary["total_logged"] == 12
    # The invariant from the test plan: no drift between log and dashboard.
    assert sum(summary["per_class_counts"].values()) == 12
    assert sum(summary["per_bin_counts"].values()) == 12


@requires_full_system
def test_routing_matches_the_threshold_for_every_item(system):
    results = system.process_batch(count=15, split="test")
    for result in results:
        if result.prediction.confidence < system.theta:
            assert result.decision.is_review is True
            assert result.decision.routed_to == HUMAN_REVIEW
        else:
            assert result.decision.is_review is False
            assert result.decision.routed_to == f"bin_{result.prediction.label}"


@requires_full_system
def test_each_item_is_actuated_to_its_decided_destination(system):
    results = system.process_batch(count=10, split="test")
    for result in results:
        assert result.actuation.destination == result.decision.routed_to
        assert len(system.actuator) == len(results)


@requires_full_system
def test_latency_stays_inside_the_one_second_requirement(system):
    results = system.process_batch(count=10, split="test")
    latencies = [result.latency_ms for result in results]
    assert max(latencies) < 1000, f"REQ-01 violated: slowest item took {max(latencies):.0f} ms"


@requires_full_system
def test_log_entry_mirrors_the_result(system):
    result = system.process_batch(count=1, split="test")[0]
    entry = result.log_entry
    assert entry.item_id == result.item_id
    assert entry.predicted_class.value == result.prediction.label
    assert entry.confidence == pytest.approx(result.prediction.confidence)
    assert entry.routed_to == result.decision.routed_to
    assert entry.is_review == result.decision.is_review
    assert entry.backend == result.prediction.backend


@requires_full_system
def test_held_out_items_carry_ground_truth(system):
    """A demo on 'test' must be a genuine held-out evaluation, not a guess."""
    results = system.process_batch(count=8, split="test")
    assert all(result.captured.true_class is not None for result in results)
    assert all(result.was_correct is not None for result in results)


@requires_full_system
def test_evaluation_accuracy_is_in_the_expected_band(system):
    """A 40-item sample should land near the recorded held-out accuracy.

    The band is deliberately wide: it catches a broken pipeline (accuracy near
    chance, ~0.17) without being flaky about which forty images were drawn.
    """
    results = system.process_batch(count=40, split="test")
    correct = sum(1 for r in results if r.was_correct)
    accuracy = correct / len(results)
    assert accuracy > 0.5, f"end-to-end accuracy {accuracy:.2f} looks broken"
    assert accuracy <= 1.0


@requires_full_system
def test_log_is_persisted_to_csv_and_reloads(system, tmp_path):
    system.process_batch(count=5, split="test")
    from ecosort.logging_db import Logger

    reloaded = Logger.from_csv(tmp_path / "log.csv")
    assert len(reloaded) == 5
    assert all(entry.predicted_class in set(WasteClass) for entry in reloaded.entries)


@requires_full_system
def test_review_items_are_excluded_from_auto_routed_accuracy(system):
    results = system.process_batch(count=30, split="test")
    auto = [r for r in results if not r.decision.is_review]
    dashboard_value = system.dashboard.auto_routed_accuracy()
    if auto:
        expected = sum(1 for r in auto if r.was_correct) / len(auto)
        assert dashboard_value == pytest.approx(expected)
    else:
        assert dashboard_value is None


@requires_full_system
def test_reset_clears_the_system(system):
    system.process_batch(count=3, split="test")
    assert len(system.logger) == 3
    system.reset()
    assert len(system.logger) == 0
    assert len(system.actuator) == 0
    assert system.dashboard.total_logged() == 0


@requires_full_system
def test_raising_the_threshold_sends_more_items_to_review(tmp_path):
    """The threshold must actually change behaviour — the whole design rests on it."""
    low = EcoSortSystem(backend="svm", theta=0.30, log_path=tmp_path / "low.csv")
    high = EcoSortSystem(backend="svm", theta=0.90, log_path=tmp_path / "high.csv")

    low.process_batch(count=20, split="test")
    high.process_batch(count=20, split="test")

    assert high.dashboard.review_count() >= low.dashboard.review_count()


@requires_full_system
def test_an_in_memory_image_keeps_its_ground_truth(system):
    """The UI sends captured items as bytes; the label must survive that trip.

    "Use a held-out sample" goes through ``process(image_bytes, true_class=...)``.
    If the bytes path drops the label, the dashboard reports no accuracy for a run
    that is a genuine evaluation — the failure this test pins down.
    """
    import io

    sample = system.capture.sample_items(count=1, split="test")[0]
    buffer = io.BytesIO()
    sample.image.save(buffer, format="PNG")

    result = system.process(buffer.getvalue(), true_class=sample.true_class)

    assert result.captured.true_class == sample.true_class
    assert result.log_entry.true_class is not None
    assert system.dashboard.accuracy_on_evaluated() is not None
