"""EcoSort — Intelligent Recycling Sorting System.

An end-to-end AI system for the point of disposal. A photo of a waste item is
captured, pre-processed, classified into one of six recycling categories, and
routed either to the matching bin or to a human-review queue when the model is
not confident enough. Every decision is logged and reflected live on a
contamination dashboard, and a simulated actuator indicates the bin.

Module layout mirrors the sub-system design in the report:

===============  =========================================================
Sub-system       Module
===============  =========================================================
Image Capture    :mod:`ecosort.capture`
Pre-processing   :mod:`ecosort.preprocess`
Waste Classifier :mod:`ecosort.classifier`   (AI sub-system)
Decision/Routing :mod:`ecosort.routing`
Logging          :mod:`ecosort.logging_db`
Dashboard        :mod:`ecosort.dashboard`
Actuation        :mod:`ecosort.actuation`
User Interface   :mod:`app` (Streamlit, at the project root)
Integration      :mod:`ecosort.pipeline`
===============  =========================================================
"""

from .classifier import MODEL_BACKENDS, Prediction, WasteClassifier
from .config import CLASS_NAMES, THETA
from .dashboard import Dashboard
from .logging_db import LogEntry, Logger, WasteClass
from .pipeline import EcoSortSystem, ItemResult
from .routing import BIN_MAP, Decision, route

__all__ = [
    "BIN_MAP",
    "CLASS_NAMES",
    "THETA",
    "MODEL_BACKENDS",
    "Dashboard",
    "Decision",
    "EcoSortSystem",
    "ItemResult",
    "LogEntry",
    "Logger",
    "Prediction",
    "WasteClass",
    "WasteClassifier",
    "route",
]

__version__ = "1.0.0"
