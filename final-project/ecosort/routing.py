"""Decision and bin-routing sub-system.

Takes a classifier output (a class label and a confidence score) and decides
what the machine does with the item: open a bin, or send the item to a person.

The threshold θ is the whole point of this sub-system. A confident item goes to
a bin. An uncertain item does not get guessed at — it is diverted to human
review, which is what makes an opaque model acceptable in a public recycling
station. θ was calibrated by sweeping the validation set (see the training
notebook and ``models/metrics.json``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .config import THETA

HUMAN_REVIEW = "human_review"

#: One physical destination per recycling category.
BIN_MAP: dict[str, str] = {
    "cardboard": "bin_cardboard",
    "glass": "bin_glass",
    "metal": "bin_metal",
    "paper": "bin_paper",
    "plastic": "bin_plastic",
    "trash": "bin_trash",
}

#: Operator-facing labels for the six destinations plus the review queue.
DESTINATION_LABELS: dict[str, str] = {
    "bin_cardboard": "Cardboard bin",
    "bin_glass": "Glass bin",
    "bin_metal": "Metal bin",
    "bin_paper": "Paper bin",
    "bin_plastic": "Plastic bin",
    "bin_trash": "General waste bin",
    HUMAN_REVIEW: "Human review queue",
}


class ClassifierOutput(Protocol):
    """Anything the router can consume.

    :class:`ecosort.classifier.Prediction` satisfies this structurally, which
    keeps the router independent of which technique produced the score. Both
    members are read-only properties so the frozen dataclass conforms.
    """

    @property
    def label(self) -> str: ...

    @property
    def confidence(self) -> float: ...


class UnknownClassError(ValueError):
    """Raised when a classifier returns a label that has no bin."""


@dataclass(frozen=True)
class Decision:
    """The routing outcome for one item."""

    predicted_class: str
    confidence: float
    routed_to: str
    is_review: bool
    theta: float

    @property
    def destination_label(self) -> str:
        """Human-readable destination, for the UI and the log."""
        return DESTINATION_LABELS.get(self.routed_to, self.routed_to)

    @property
    def reasoning(self) -> str:
        """One sentence explaining the decision, using the actual numbers."""
        if self.is_review:
            # Naming the queue itself here read as "instead of the review queue".
            # What the reviewer needs to know is which bin was avoided.
            would_be_bin = BIN_MAP.get(self.predicted_class)
            avoided = (DESTINATION_LABELS[would_be_bin].lower()
                       if would_be_bin else "automatic bin")
            return (
                f"Confidence {self.confidence:.0%} is below the {self.theta:.0%} threshold, "
                f"so the item goes to a person rather than straight to the {avoided}."
            )
        return (
            f"Confidence {self.confidence:.0%} meets the {self.theta:.0%} threshold, "
            f"so the item goes to the {self.destination_label.lower()}."
        )


def bin_for(predicted_class: str) -> str:
    """Return the bin name for a class, or raise if the class is unknown."""
    try:
        return BIN_MAP[predicted_class]
    except KeyError as exc:
        raise UnknownClassError(
            f"No bin mapped for class '{predicted_class}'. "
            f"Known classes: {', '.join(sorted(BIN_MAP))}"
        ) from exc


def route(
    prediction: ClassifierOutput,
    theta: float | None = None,
    bin_map: Mapping[str, str] | None = None,
) -> Decision:
    """Route one item using the confidence threshold θ.

    Args:
        prediction: Classifier output carrying ``label`` and ``confidence``.
        theta: Confidence threshold. Defaults to the calibrated value.
        bin_map: Optional alternative destination mapping.

    Returns:
        A :class:`Decision` naming the destination and the reason.
    """
    threshold = THETA if theta is None else float(theta)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"theta must be within [0, 1], got {threshold}")

    confidence = float(prediction.confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0, 1], got {confidence}")

    mapping = bin_map or BIN_MAP
    label = str(prediction.label)

    if confidence < threshold:
        return Decision(
            predicted_class=label,
            confidence=confidence,
            routed_to=HUMAN_REVIEW,
            is_review=True,
            theta=threshold,
        )

    try:
        destination = mapping[label]
    except KeyError as exc:
        raise UnknownClassError(
            f"No bin mapped for class '{label}'. Known classes: {', '.join(sorted(mapping))}"
        ) from exc

    return Decision(
        predicted_class=label,
        confidence=confidence,
        routed_to=destination,
        is_review=False,
        theta=threshold,
    )
