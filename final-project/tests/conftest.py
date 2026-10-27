"""Shared fixtures for the EcoSort test suite.

Tests that need trained artefacts or the dataset skip cleanly when those are
absent, so the suite is meaningful in a fresh clone and complete in a trained
one.

The project root is prepended to ``sys.path`` below, which is the standard
pytest ``conftest`` pattern for running the suite without an install step. That
makes every import after it technically "not at top of file", so the rule is
disabled for the whole module rather than sprinkled line by line.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for _entry in (str(PROJECT_ROOT), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import pytest

from ecosort.capture import ImageCapture

from .helpers import (
    data_available,
    full_system_available,
    model_available,
    split_available,
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def capture() -> ImageCapture:
    return ImageCapture()


__all__ = [
    "PROJECT_ROOT",
    "capture",
    "data_available",
    "full_system_available",
    "model_available",
    "project_root",
    "split_available",
]
