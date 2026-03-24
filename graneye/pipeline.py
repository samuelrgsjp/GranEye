from __future__ import annotations

from collections.abc import Iterable

from .analyzers.base import Analyzer
from .clustering import cluster_identities
from .detection import is_directory_url
from .models import AnalysisResult, IdentityCluster, ProfileRecord


def analyze_records(
    records: Iterable[ProfileRecord],
    analyzer: Analyzer,
) -> list[tuple[ProfileRecord, AnalysisResult]]:
    """Run analyzer on non-directory records only."""

    results: list[tuple[ProfileRecord, AnalysisResult]] = []
    for record in records:
        if is_directory_url(record.url):
            continue
        results.append((record, analyzer.analyze(record)))
    return results


def cluster_records(records: Iterable[ProfileRecord]) -> list[IdentityCluster]:
    """Public clustering façade kept separate for testability."""

    return cluster_identities(records)
