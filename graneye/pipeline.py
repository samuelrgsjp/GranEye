from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .analyzers.base import Analyzer
from .clustering import cluster_identities
from .detection import is_directory_url
from .models import AnalysisResult, IdentityCluster, ProfileRecord
from .resolution import ContextQuery, ResolutionOutput, ScoredCandidate, rank_candidates, resolve_identity
from .search import (
    FilterDecision,
    SearchResult,
    filter_search_results,
    normalize_search_results,
)


@dataclass(slots=True, frozen=True)
class SearchPipelineDiagnostics:
    raw_results_count: int
    normalized_results_count: int
    filtered_results_count: int
    ranked_candidates_count: int
    filter_decisions: tuple[FilterDecision, ...]
    ranked_candidates: tuple[ScoredCandidate, ...]
    query_attempts: tuple[str, ...] = ()
    source_diversity_count: int = 0
    ambiguity_triggered: bool = False
    ambiguity_reason: str = ""


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

    # Allow natural forms such as "Software Engineer in London" and "CEO at Google".
    phrase_match = re.match(
        r"^(?P<profession>.+?)\s+(?:in|at|of|from)\s+(?P<location>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80})$",
        normalized,
        flags=re.IGNORECASE,
    )
    if phrase_match:
        profession = phrase_match.group("profession").strip()
        location = phrase_match.group("location").strip()
        if profession and location:
            return profession, location

    return normalized, None


def _query_variants(target_name: str, context: str | None) -> list[str]:
    base = target_name.strip()
    if not context:
        return [base, f"\"{base}\"", f"{base} profile", f"{base} biography"]
    context_clean = context.strip()
    if not context_clean:
        return [base]
    variants = [
        f"{base} {context_clean}",
        f"\"{base}\" {context_clean}",
        f"{base} {context_clean} profile",
        f"{base} {context_clean} biography",
        f"\"{base}\" {context_clean} site:.edu",
        f"{base} {context_clean} linkedin",
        base,
        f"\"{base}\"",
        f"{base} profile",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in variants:
        normalized = " ".join(item.split()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _run_search(
    query_texts: list[str],
    *,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> tuple[list[Mapping[str, str]], tuple[str, ...]]:
    combined_results: list[Mapping[str, str]] = []
    seen_urls: set[str] = set()
    seen_signatures: set[tuple[str, str]] = set()
    attempted_queries: list[str] = []
    for query_text in query_texts:
        attempted_queries.append(query_text)
        try:
            raw_results = html_search(query_text)
        except Exception:
            raw_results = []
        try:
            instant_results = instant_search(query_text)
        except Exception:
            instant_results = []

        merged_sources = [*raw_results, *instant_results]
        if len(merged_sources) <= 1 and " " in query_text:
            # Name-only fallback for weak one-result query outcomes.
            quoted = re.findall(r'"([^"]+)"', query_text)
            if quoted:
                name_only = quoted[0].strip()
            else:
                tokens = query_text.split()
                name_only = " ".join(tokens[:2]).strip()
            if len(name_only) >= 3:
                try:
                    merged_sources.extend(html_search(name_only))
                except Exception:
                    pass

        for item in merged_sources:
            url = str(item.get("url") or item.get("link")).strip()
            canonical_url = _canonicalize_url(url)
            title_signature = " ".join(str(item.get("title") or "").casefold().split())
            signature = (title_signature, urlparse(canonical_url).netloc.casefold())
            if canonical_url and canonical_url in seen_urls:
                continue
            if signature in seen_signatures:
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
            if signature != ("", ""):
                seen_signatures.add(signature)
            combined_results.append(item)
        if len(combined_results) >= 12:
            break
    return combined_results, tuple(attempted_queries)


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    cleaned = parsed._replace(
        netloc=parsed.netloc.casefold().removeprefix("www."),
        query=urlencode(query_pairs, doseq=True),
        fragment="",
        path=parsed.path.rstrip("/") or "/",
    )
    return urlunparse(cleaned)


def resolve_query(
    target_name: str,
    *,
    context: str | None,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> tuple[ResolutionOutput | None, list[ScoredCandidate]]:
    """End-to-end deterministic orchestration from query to ranked candidate output."""

    combined_results, _ = _run_search(
        _query_variants(target_name, context),
        html_search=html_search,
        instant_search=instant_search,
    )

    normalized_results = normalize_search_results(combined_results)
    search_results, _ = filter_search_results(normalized_results)
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


def resolve_query_with_debug(
    target_name: str,
    *,
    context: str | None,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> tuple[ResolutionOutput | None, list[ScoredCandidate], SearchPipelineDiagnostics]:
    combined_results, query_attempts = _run_search(
        _query_variants(target_name, context),
        html_search=html_search,
        instant_search=instant_search,
    )

    normalized_results = normalize_search_results(combined_results)
    search_results, filter_decisions = filter_search_results(normalized_results)
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

    diagnostics = SearchPipelineDiagnostics(
        raw_results_count=len(combined_results),
        normalized_results_count=len(normalized_results),
        filtered_results_count=len(search_results),
        ranked_candidates_count=len(ranked),
        filter_decisions=tuple(filter_decisions),
        ranked_candidates=tuple(ranked),
        query_attempts=query_attempts,
        source_diversity_count=len({item.result.domain for item in ranked}),
        ambiguity_triggered=bool(resolved and resolved.ambiguity_detected),
        ambiguity_reason=resolved.ambiguity_reason or "" if resolved else "",
    )
    return resolved, ranked, diagnostics
