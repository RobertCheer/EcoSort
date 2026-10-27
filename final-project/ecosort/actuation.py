"""Simulated actuation sub-system.

A physical EcoSort station opens a bin flap and lights an indicator. This
module reproduces that behaviour without hardware, so the end-to-end demo shows
the same sequence a real station would perform, and so the actuation step is a
testable object rather than a print statement buried in the UI.

The state machine is deliberately small: exactly one destination is active at
a time, and an item that goes to human review lights the review indicator
instead of opening a bin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .routing import DESTINATION_LABELS, HUMAN_REVIEW, Decision

#: Indicator colour per destination, following recycling-station conventions.
BIN_COLOURS: dict[str, str] = {
    "bin_cardboard": "#8D6E63",
    "bin_glass": "#2E7D32",
    "bin_metal": "#607D8B",
    "bin_paper": "#1565C0",
    "bin_plastic": "#F9A825",
    "bin_trash": "#616161",
    HUMAN_REVIEW: "#C62828",
}

#: Short operator-facing action text, as the station would display it.
ACTION_TEXT: dict[str, str] = {
    "bin_cardboard": "Open cardboard flap",
    "bin_glass": "Open glass flap",
    "bin_metal": "Open metal flap",
    "bin_paper": "Open paper flap",
    "bin_plastic": "Open plastic flap",
    "bin_trash": "Open general-waste flap",
    HUMAN_REVIEW: "Divert to review tray, hold item",
}


@dataclass(frozen=True)
class ActuationState:
    """The physical result of one decision."""

    destination: str
    colour: str
    action: str
    label: str
    is_review: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def display_label(self) -> str:
        return DESTINATION_LABELS.get(self.destination, self.destination)


class SimulatedActuator:
    """Simulated bin mechanism, recording what it was told to do."""

    def __init__(self) -> None:
        self._history: list[ActuationState] = []
        self._active: ActuationState | None = None

    @property
    def active(self) -> ActuationState | None:
        """The destination currently held open, if any."""
        return self._active

    @property
    def history(self) -> list[ActuationState]:
        return list(self._history)

    def activate(self, decision: Decision) -> ActuationState:
        """Perform the physical response for one routed decision."""
        destination = decision.routed_to
        state = ActuationState(
            destination=destination,
            colour=BIN_COLOURS.get(destination, "#616161"),
            action=ACTION_TEXT.get(destination, f"Open {destination}"),
            label=DESTINATION_LABELS.get(destination, destination),
            is_review=decision.is_review,
        )
        self._active = state
        self._history.append(state)
        return state

    def reset(self) -> None:
        """Close every flap, as the station does once an item has dropped."""
        self._active = None

    def clear_history(self) -> None:
        self._history.clear()
        self._active = None

    def __len__(self) -> int:
        return len(self._history)
