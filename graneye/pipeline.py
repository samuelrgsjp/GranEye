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
    context_interpretation: str = ""


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


_ROLE_HINTS = {
    "engineer",
    "developer",
    "lawyer",
    "avocat",
    "psychologist",
    "professor",
    "profesora",
    "actor",
    "actress",
    "musician",
    "artist",
    "journalist",
    "influencer",
    "creator",
    "streamer",
    "politician",
    "athlete",
    "ceo",
    "executive",
    "author",
    "speaker",
    "host",
}
_PLATFORM_HINTS = {"youtube", "twitch", "tiktok", "instagram", "x", "twitter", "linkedin", "github", "wikipedia"}
_INSTITUTIONAL_HINTS = {"university", "faculty", "department", "staff", "official", "government", "ministerio"}
_LOCATION_PREPOSITIONS = {"in", "en", "de", "from", "at"}
_STOP_TOKENS = {"the", "a", "an", "and", "of", "del", "la", "el"}


def _parse_context(context: str | None) -> ContextQuery:
    if context is None:
        return ContextQuery()

    normalized = context.strip()
    if not normalized:
        return ContextQuery()

    role: str | None = None
    location: str | None = None
    organization: str | None = None
    media_platform: str | None = None
    institutional_hint: str | None = None
    domain_activity: str | None = None

    tokens = [token for token in re.findall(r"[A-Za-zÀ-ÿ0-9]+", normalized.casefold()) if token]
    token_set = set(tokens)
    role_tokens = [token for token in tokens if token in _ROLE_HINTS]
    platform_tokens = [token for token in tokens if token in _PLATFORM_HINTS]
    institutional_tokens = [token for token in tokens if token in _INSTITUTIONAL_HINTS]

    if role_tokens:
        role = " ".join(dict.fromkeys(role_tokens))
    if platform_tokens:
        media_platform = " ".join(dict.fromkeys(platform_tokens))
    if institutional_tokens:
        institutional_hint = " ".join(dict.fromkeys(institutional_tokens))

    phrase_match = re.match(
        r"^(?P<left>.+?)\s+(?:in|en|de|from|at|for)\s+(?P<right>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80})$",
        normalized,
        flags=re.IGNORECASE,
    )
    if phrase_match:
        left = phrase_match.group("left").strip()
        right = phrase_match.group("right").strip()
        if left and right:
            if role is None and any(token in _ROLE_HINTS for token in re.findall(r"[A-Za-zÀ-ÿ]+", left.casefold())):
                role = left
            elif organization is None:
                organization = left
            location = right
    elif "," in normalized:
        left, right = [part.strip() for part in normalized.split(",", 1)]
        if left and right:
            role = role or left
            location = right

    if location is None:
        cap_tokens = re.findall(r"\b[A-ZÀ-Ý][a-zà-ÿ'-]{2,}\b", normalized)
        if cap_tokens and len(cap_tokens) <= 3:
            candidate = " ".join(cap_tokens)
            lowered_parts = set(re.findall(r"[A-Za-zÀ-ÿ]+", candidate.casefold()))
            if not lowered_parts & (_ROLE_HINTS | _PLATFORM_HINTS | _INSTITUTIONAL_HINTS):
                location = candidate

    if organization is None:
        org_match = re.search(r"\b(?:at|for|of|de)\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9&.\- ]{2,60})", normalized)
        if org_match:
            organization = org_match.group(1).strip()

    generic_terms = tuple(
        token for token in dict.fromkeys(tokens) if token not in _STOP_TOKENS and token not in _LOCATION_PREPOSITIONS
    )
    if normalized and len(tokens) <= 4 and not any(char.isdigit() for char in normalized):
        domain_activity = normalized

    return ContextQuery(
        role=role,
        organization=organization,
        location=location,
        domain_activity=domain_activity,
        media_platform=media_platform,
        institutional_hint=institutional_hint,
        raw_context=normalized,
        generic_terms=generic_terms,
    )


def _query_variants(target_name: str, context: str | None) -> list[str]:
    base = target_name.strip()
    if not context:
        return [base, f"\"{base}\"", f"{base} profile", f"{base} biography", f"{base} official", f"{base} public profile"]
    context_data = _parse_context(context)
    context_clean = context.strip() if context else ""
    if not context_clean:
        return [base, f"\"{base}\""]
    hint_fragments = [context_data.role, context_data.organization, context_data.location, context_data.media_platform, context_data.institutional_hint]
    compact_hints = " ".join(part for part in hint_fragments if part).strip()
    variants = [
        f"{base} {context_clean}",
        f"\"{base}\" {context_clean}",
        f"{base} {context_clean} profile",
        f"{base} {context_clean} biography",
        f"\"{base}\" {context_clean} site:.edu",
        f"{base} {context_clean} official",
        f"{base} {context_clean} public profile",
        f"\"{base}\" {context_clean} official bio",
        f"{base} {context_clean} linkedin",
        f"{base} {context_clean} wikipedia",
        base,
        f"\"{base}\"",
        f"{base} profile",
    ]
    if compact_hints:
        variants.extend(
            [
                f"{base} {compact_hints}",
                f"\"{base}\" {compact_hints} official bio",
                f"{base} {compact_hints} interview",
            ]
        )
    if context_data.media_platform:
        variants.append(f"{base} {context_data.media_platform} channel")
    if context_data.institutional_hint:
        variants.append(f"{base} {context_data.institutional_hint} staff")
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
    parsed_context = _parse_context(context)
    ranked = rank_candidates(
        search_results,
        target_name,
        parsed_context,
    )

    try:
        resolved = resolve_identity(
            target_name,
            search_results,
            role=parsed_context.role,
            organization=parsed_context.organization,
            location=parsed_context.location,
            domain_activity=parsed_context.domain_activity,
            media_platform=parsed_context.media_platform,
            institutional_hint=parsed_context.institutional_hint,
            raw_context=parsed_context.raw_context,
            generic_terms=parsed_context.generic_terms,
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
    parsed_context = _parse_context(context)
    ranked = rank_candidates(
        search_results,
        target_name,
        parsed_context,
    )

    try:
        resolved = resolve_identity(
            target_name,
            search_results,
            role=parsed_context.role,
            organization=parsed_context.organization,
            location=parsed_context.location,
            domain_activity=parsed_context.domain_activity,
            media_platform=parsed_context.media_platform,
            institutional_hint=parsed_context.institutional_hint,
            raw_context=parsed_context.raw_context,
            generic_terms=parsed_context.generic_terms,
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
        context_interpretation=(
            f"role={parsed_context.role or '-'}; org={parsed_context.organization or '-'}; "
            f"location={parsed_context.location or '-'}; activity={parsed_context.domain_activity or '-'}; "
            f"platform={parsed_context.media_platform or '-'}; institutional={parsed_context.institutional_hint or '-'}; "
            f"generic={','.join(parsed_context.generic_terms[:6]) if parsed_context.generic_terms else '-'}"
        ),
    )
    return resolved, ranked, diagnostics
