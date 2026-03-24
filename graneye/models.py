from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Candidate:
    """Identity candidate extracted from free-form text."""

    raw: str
    normalized: str
    confidence: float
    source: str


@dataclass(slots=True, frozen=True)
class ProfileRecord:
    """Single OSINT profile-like record."""

    identifier: str
    display_name: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class IdentityCluster:
    """Collection of records likely belonging to the same identity."""

    key: str
    members: tuple[ProfileRecord, ...]
    confidence: float


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    """Result returned by analyzer implementations."""

    risk_score: float
    tags: tuple[str, ...]
    rationale: str
