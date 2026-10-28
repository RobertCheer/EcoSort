"""Decision and bin-routing tests.

These cover the REQ-01 routing clause directly: a class at or above the
threshold opens its bin, anything below goes to a person, and no class can be
routed without a destination.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from ecosort.routing import (
    BIN_MAP,
    DESTINATION_LABELS,
    HUMAN_REVIEW,
    Decision,
    UnknownClassError,
    bin_for,
    route,
)


@dataclass(frozen=True)
class StubPrediction:
    label: str
    confidence: float


def test_confident_item_routes_to_its_bin():
    decision = route(StubPrediction("glass", 0.91), theta=0.60)
    assert decision.routed_to == "bin_glass"
    assert decision.is_review is False
    assert decision.predicted_class == "glass"


def test_low_confidence_item_goes_to_human_review():
    decision = route(StubPrediction("glass", 0.42), theta=0.60)
    assert decision.routed_to == HUMAN_REVIEW
    assert decision.is_review is True


def test_confidence_exactly_at_threshold_is_accepted():
    """The boundary is inclusive: θ itself is confident enough."""
    decision = route(StubPrediction("paper", 0.60), theta=0.60)
    assert decision.is_review is False
    assert decision.routed_to == "bin_paper"


def test_just_below_threshold_is_reviewed():
    decision = route(StubPrediction("paper", 0.5999), theta=0.60)
    assert decision.is_review is True


@pytest.mark.parametrize("class_name", sorted(BIN_MAP))
def test_every_class_has_a_bin(class_name):
    decision = route(StubPrediction(class_name, 0.99), theta=0.60)
    assert decision.routed_to == BIN_MAP[class_name]
    assert decision.destination_label in DESTINATION_LABELS.values()


def test_every_bin_has_an_operator_label():
    for destination in list(BIN_MAP.values()) + [HUMAN_REVIEW]:
        assert DESTINATION_LABELS.get(destination), f"no label for {destination}"


def test_unknown_class_raises_instead_of_silently_routing():
    with pytest.raises(UnknownClassError):
        route(StubPrediction("compost", 0.99), theta=0.60)


def test_unknown_class_below_threshold_is_still_reviewed_not_raised():
    """An unknown label at low confidence is safe: a person sees it anyway."""
    decision = route(StubPrediction("compost", 0.10), theta=0.60)
    assert decision.is_review is True
    assert decision.routed_to == HUMAN_REVIEW


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        route(StubPrediction("glass", 0.9), theta=1.4)


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError):
        route(StubPrediction("glass", 1.4), theta=0.6)


def test_decision_is_frozen():
    decision = route(StubPrediction("metal", 0.8), theta=0.6)
    with pytest.raises(FrozenInstanceError):
        decision.routed_to = "bin_glass"  # type: ignore[misc]


def test_reasoning_quotes_the_real_numbers():
    accepted = route(StubPrediction("metal", 0.80), theta=0.60)
    assert "80%" in accepted.reasoning and "60%" in accepted.reasoning
    reviewed = route(StubPrediction("metal", 0.30), theta=0.60)
    assert "30%" in reviewed.reasoning and "60%" in reviewed.reasoning


def test_bin_for_maps_and_raises():
    assert bin_for("plastic") == "bin_plastic"
    with pytest.raises(UnknownClassError):
        bin_for("nope")


def test_destination_label_for_review_queue():
    decision = route(StubPrediction("trash", 0.1), theta=0.6)
    assert "review" in decision.destination_label.lower()
    assert isinstance(decision, Decision)
