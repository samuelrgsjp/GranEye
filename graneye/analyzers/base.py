from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import AnalysisResult, ProfileRecord


class Analyzer(Protocol):
    """Stable analyzer interface used by the OSINT pipeline."""

    def analyze(self, record: ProfileRecord) -> AnalysisResult:
        """Return an analysis result for a single profile record."""


@dataclass(slots=True)
class LocalAnalyzerInput:
    """Structured hand-off payload for future local analyzers."""

    identifier: str
    display_name: str
    url: str
    context: dict[str, str]


class LocalAnalyzerBackend(Protocol):
    """Future-compatible backend contract (e.g., local ML model)."""

    def run(self, payload: LocalAnalyzerInput) -> AnalysisResult:
        """Analyze a pre-normalized payload and return a rule-compatible result."""
