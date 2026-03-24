from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .analyzers.base import Analyzer
from .clustering import cluster_identities
from .detection import is_directory_url
from .models import AnalysisResult, IdentityCluster, ProfileRecord
from .resolution import ContextQuery, ResolutionOutput, ScoredCandidate, rank_candidates, resolve_identity
from .search import SearchResult, enrich_search_results


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


def _context_parts(context: str | None) -> tuple[str | None, str | None]:
    if context is None:
        return None, None

    normalized = context.strip()
    if not normalized:
        return None, None

    if "," in normalized:
        profession, location = [part.strip() for part in normalized.split(",", 1)]
        return profession or None, location or None

    return normalized, None


def resolve_query(
    target_name: str,
    *,
    context: str | None,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> tuple[ResolutionOutput | None, list[ScoredCandidate]]:
    """End-to-end deterministic orchestration from query to ranked candidate output."""

    query_text = " ".join(part for part in [target_name.strip(), (context or "").strip()] if part)
    raw_results: list[Mapping[str, str]] = []
    instant_results: list[Mapping[str, str]] = []
    try:
        raw_results = html_search(query_text)
    except Exception:
        raw_results = []

    try:
        instant_results = instant_search(query_text)
    except Exception:
        instant_results = []

    combined_results = [*raw_results]
    if instant_results:
        seen = {
            str(item.get("url") or item.get("link")).strip()
            for item in raw_results
            if str(item.get("url") or item.get("link")).strip()
        }
        for item in instant_results:
            url = str(item.get("url") or item.get("link")).strip()
            if url and url not in seen:
                combined_results.append(item)
                seen.add(url)

    search_results = enrich_search_results(combined_results)
    profession, location = _context_parts(context)
    ranked = rank_candidates(
        search_results,
        target_name,
        ContextQuery(profession=profession, location=location),
    )

    try:
        resolved = resolve_identity(
            target_name,
            search_results,
            profession=profession,
            location=location,
        )
    except Exception:
        resolved = None

    return resolved, ranked
