"""Image Capture sub-system.

Normalises the three ways an item photo can arrive at the station — a file
upload, a webcam frame, or a scheduled path — into one :class:`CapturedItem`.

It also provides *sample items* drawn from the held-out test split. That is a
demonstration aid, not a shortcut: the samples are the same 380 images the
classifier was never trained on, so a demo run is a genuine held-out test and
the dashboard's contamination rate means something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_DIR, MODEL_DIR, class_names
from .preprocess import InvalidImageError, load_image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _read_json(path: Path) -> dict:
    """Read a JSON mapping, reporting a corrupt file clearly instead of raising raw."""
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InvalidImageError(f"Could not parse {path}: {exc}") from exc


@dataclass(frozen=True)
class CapturedItem:
    """One item presented to the station."""

    item_id: str
    image: Image.Image
    source: str
    #: File path when the photo came from disk. The review queue uses it to show
    #: the item again after a restart, when no in-memory preview exists.
    source_path: str | None = None
    true_class: str | None = None
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def has_ground_truth(self) -> bool:
        return self.true_class is not None


class ImageCapture:
    """Acquires item photos and labels them by origin."""

    def __init__(
        self,
        data_dir: Path | str = DATA_DIR,
        split_path: Path | str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split_path = Path(split_path) if split_path else MODEL_DIR / "split_indices.json"
        self._index_cache: list[tuple[Path, str]] | None = None
        self._counter = 0

    # --------------------------------------------------------------- capture

    def _next_id(self) -> str:
        self._counter += 1
        return f"item_{self._counter:04d}"

    def from_upload(
        self,
        data: bytes,
        item_id: str | None = None,
        true_class: str | None = None,
    ) -> CapturedItem:
        """Capture from an uploaded file or an in-memory image (the UI path).

        ``true_class`` is not decoration: the UI sends held-out samples through
        this path, and dropping the label there would make the dashboard report
        no accuracy at all for a run that is a genuine evaluation.
        """
        return CapturedItem(
            item_id=item_id or self._next_id(),
            image=load_image(data),
            source="upload",
            true_class=true_class,
        )

    def from_path(
        self,
        path: Path | str,
        item_id: str | None = None,
        true_class: str | None = None,
    ) -> CapturedItem:
        """Capture from a file already on disk (batch and test path)."""
        return CapturedItem(
            item_id=item_id or self._next_id(),
            image=load_image(path),
            source="path",
            true_class=true_class,
            source_path=str(Path(path)),
        )

    def from_frame(self, frame: np.ndarray | Image.Image, item_id: str | None = None, true_class: str | None = None) -> CapturedItem:
        """Capture from a webcam frame (a numpy array or PIL image)."""
        return CapturedItem(
            item_id=item_id or self._next_id(),
            image=load_image(frame),
            source="camera",
            true_class=true_class,
        )

    # ---------------------------------------------------------------- samples

    def _build_index(self) -> list[tuple[Path, str]]:
        """Rebuild the exact image order used during training.

        Training walked the class folders in sorted order and each folder's
        filenames in sorted order. Reproducing that order is what lets the saved
        split indices refer to the same images here.
        """
        if self._index_cache is not None:
            return self._index_cache

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.data_dir}")

        index: list[tuple[Path, str]] = []
        for label in class_names():
            class_dir = self.data_dir / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for filename in sorted(p.name for p in class_dir.iterdir()):
                if filename.startswith("._") or not filename.lower().endswith(IMAGE_SUFFIXES):
                    continue
                index.append((class_dir / filename, label))

        self._index_cache = index
        return index

    def _split_indices(self, split: str) -> list[int]:
        if not self.split_path.is_file():
            raise FileNotFoundError(
                f"Split indices not found at {self.split_path}. "
                "Run the training notebook to generate them."
            )
        payload = _read_json(self.split_path)
        if split not in payload:
            raise KeyError(f"Split '{split}' missing from {self.split_path}")
        return list(payload[split])

    def available_samples(self, split: str = "test") -> int:
        """How many sample items this split can supply."""
        self._build_index()
        return len(self._split_indices(split))

    def sample_items(
        self,
        count: int = 1,
        split: str = "test",
        class_name: str | None = None,
        start: int = 0,
    ) -> list[CapturedItem]:
        """Draw held-out items for a demonstration run.

        Args:
            count: how many items to draw.
            split: ``train``, ``val`` or ``test``.
            class_name: restrict to one category, or ``None`` for any.
            start: offset into the split, so successive calls walk new items
                instead of repeating the same few.

        Returns:
            Captured items carrying their true class, so evaluation is real.
        """
        index = self._build_index()
        positions = self._split_indices(split)
        if class_name is not None and class_name not in class_names():
            raise ValueError(f"Unknown class '{class_name}'. Known: {', '.join(class_names())}")

        chosen: list[CapturedItem] = []
        for offset in range(start, len(positions)):
            path, label = index[positions[offset]]
            if class_name is not None and label != class_name:
                continue
            chosen.append(self.from_path(path, true_class=label))
            if len(chosen) >= count:
                break

        if not chosen:
            raise InvalidImageError(
                f"No sample items available for split='{split}', class='{class_name}', start={start}"
            )
        return chosen
