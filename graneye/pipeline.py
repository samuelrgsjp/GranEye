from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .analyzers.base import Analyzer
from .clustering import cluster_identities
from .detection import is_directory_url
from .models import AnalysisResult, IdentityCluster, ProfileRecord
from .resolution import (
    ContextQuery,
    ResolutionOutput,
    ScoredCandidate,
    assess_query_validity,
    rank_candidates,
    resolve_identity,
)
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
    query_validity: str = "valid"


@dataclass(slots=True, frozen=True)
class _PipelineExecution:
    combined_results: tuple[Mapping[str, str], ...]
    query_attempts: tuple[str, ...]
    normalized_results_count: int
    search_results: tuple[SearchResult, ...]
    filter_decisions: tuple[FilterDecision, ...]
    ranked: tuple[ScoredCandidate, ...]
    parsed_context: ContextQuery
    resolved: ResolutionOutput | None
    query_validity: str


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
    "singer",
    "footballer",
    "player",
    "founder",
    "cofounder",
    "co-founder",
    "administrator",
    "administrador",
    "sysadmin",
    "support",
    "soporte",
    "informatico",
    "informática",
    "informatica",
    "it",
    "systems",
    "sistemas",
}
_PLATFORM_HINTS = {"youtube", "twitch", "tiktok", "instagram", "x", "twitter", "linkedin", "github", "wikipedia"}
_INSTITUTIONAL_HINTS = {"university", "faculty", "department", "staff", "official", "government", "ministerio"}
_LOCATION_PREPOSITIONS = {"in", "en", "de", "from", "at"}
_STOP_TOKENS = {"the", "a", "an", "and", "of", "del", "la", "el"}
_KNOWN_LOCATION_TOKENS = {
    "spain",
    "madrid",
    "london",
    "seattle",
    "austin",
    "boston",
    "barcelona",
    "valencia",
    "paris",
    "berlin",
    "rome",
    "argentina",
    "usa",
    "cantabria",
    "españa",
    "espana",
}
_ACTIVITY_HINTS = {"cybersecurity", "security", "software", "cloud", "finance", "healthcare", "ai", "ml", "data"}
_ORG_STOPWORDS = _ROLE_HINTS | {
    "founder",
    "cofounder",
    "co-founder",
    "chairman",
    "music",
    "football",
    "youtube",
    "system",
    "sistemas",
}
_ROLE_PHRASES = (
    "software engineer",
    "platform engineering director",
    "data engineer",
    "machine learning engineer",
    "security engineer",
    "system administrator",
    "it administrator",
    "it support",
    "soporte it",
    "tecnico it",
    "técnico it",
    "informatico",
    "informático",
    "administrador de sistemas",
)
_ROLE_TOKEN_EXCLUSIONS = {"en", "in"}
_ORG_VERB_PREFIXES = ("trabaja en ", "works at ", "employee at ", "empleado en ")
_ORG_LINKING_STOPWORDS = {
    "trabaja",
    "trabajar",
    "works",
    "work",
    "employee",
    "empleado",
    "empleada",
    "en",
    "in",
    "de",
    "del",
    "at",
    "for",
    "with",
    "of",
}
_PRIVATE_PROFILE_MARKERS = (
    "linkedin",
    "staff",
    "team",
    "profile",
    "people",
    "about",
    "equipo",
    "plantilla",
    "personal",
    "perfil",
    "empresa",
)


def _private_person_context_strength(context_data: ContextQuery) -> int:
    return sum(
        1
        for value in (
            context_data.role,
            context_data.organization,
            context_data.location,
            context_data.domain_activity,
        )
        if value
    )


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

    words = [token for token in re.findall(r"[A-Za-zÀ-ÿ0-9]+", normalized) if token]
    tokens = [token.casefold() for token in words]

    lowered = " ".join(tokens)

    def _clean_fragment(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" ,.;:-")
        cleaned = re.sub(r"\b(?:en|in)\s*$", "", cleaned, flags=re.I).strip(" ,.;:-")
        return cleaned

    def _dedupe_tokens(value: str) -> str:
        parts = [part for part in re.findall(r"[A-Za-zÀ-ÿ0-9]+", value) if part]
        ordered = [part for part in dict.fromkeys(parts)]
        return " ".join(ordered).strip()

    def _looks_role_like(value: str | None) -> bool:
        if not value:
            return False
        value_tokens = [token.casefold() for token in re.findall(r"[A-Za-zÀ-ÿ]+", value)]
        if not value_tokens:
            return False
        return any(token in _ROLE_HINTS for token in value_tokens)
    for phrase in _ROLE_PHRASES:
        if phrase in lowered:
            role = phrase
            break
    if role is None:
        role_tokens = [token for token in tokens if token in _ROLE_HINTS]
        if role_tokens:
            role = " ".join(dict.fromkeys(role_tokens))
    if role is not None:
        role = " ".join(token for token in role.split() if token not in _ROLE_TOKEN_EXCLUSIONS).strip()
        if role == "sysadmin":
            role = "system administrator"

    platform_tokens = [token for token in tokens if token in _PLATFORM_HINTS]
    if platform_tokens:
        media_platform = " ".join(dict.fromkeys(platform_tokens))
    institutional_tokens = [token for token in tokens if token in _INSTITUTIONAL_HINTS]
    if institutional_tokens:
        institutional_hint = " ".join(dict.fromkeys(institutional_tokens))

    org_loc_pattern = re.search(
        r"\b(?:trabaja en|works at|employee at|empleado en)\s+([A-Za-zÀ-ÿ0-9&.'-][A-Za-zÀ-ÿ0-9&.'\- ]{1,80}?)\s+(?:en|in)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80})$",
        normalized,
        flags=re.I,
    )
    if org_loc_pattern:
        organization = _clean_fragment(org_loc_pattern.group(1))
        location = _clean_fragment(org_loc_pattern.group(2))

    org_only_pattern = re.search(
        r"\b(?:trabaja en|works at|employee at|empleado en)\s+([A-Za-zÀ-ÿ0-9&.'-][A-Za-zÀ-ÿ0-9&.'\- ]{1,80})$",
        normalized,
        flags=re.I,
    )
    if org_only_pattern and organization is None:
        organization = _clean_fragment(org_only_pattern.group(1))

    prep_loc = re.search(r"\b(?:in|en|from|based in|located in)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,80})$", normalized, flags=re.I)
    if prep_loc:
        role_loc_candidate = normalized[: prep_loc.start()].strip(" ,.;:-")
        if role and role_loc_candidate and any(token in role_loc_candidate.casefold() for token in role.split()):
            location = _clean_fragment(prep_loc.group(1))
        elif role is None and _looks_role_like(role_loc_candidate):
            role = _dedupe_tokens(role_loc_candidate.casefold())
            if role == "sysadmin":
                role = "system administrator"
            location = _clean_fragment(prep_loc.group(1))
        elif location is None:
            location = _clean_fragment(prep_loc.group(1))
    elif tokens and tokens[-1] in _KNOWN_LOCATION_TOKENS:
        location = words[-1]
    elif any(token in _KNOWN_LOCATION_TOKENS for token in tokens):
        idx = next(i for i, token in enumerate(tokens) if token in _KNOWN_LOCATION_TOKENS)
        location = words[idx]
    elif "," in normalized:
        left, right = [part.strip() for part in normalized.split(",", 1)]
        if right:
            location = right
        if left and role is None:
            role = left.casefold()

    org_match = re.search(r"\b(?:at|for|with|of|de)\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9&.\- ]{2,60})", normalized)
    if org_match and organization is None:
        candidate_org = _clean_fragment(org_match.group(1))
        if candidate_org and not _looks_role_like(candidate_org):
            organization = candidate_org

    if organization is None and role and len(words) >= 2:
        role_tokens = set(role.split())
        role_positions = [idx for idx, token in enumerate(tokens) if token in role_tokens]
        if role_positions:
            first_role_idx = role_positions[0]
            prefix = words[:first_role_idx]
            if prefix and any(piece[:1].isupper() for piece in prefix) and all(
                piece.casefold() not in _KNOWN_LOCATION_TOKENS for piece in prefix
            ):
                organization = " ".join(prefix).strip()
    if organization is None and len(words) == 2 and tokens[1] in _KNOWN_LOCATION_TOKENS and role is None:
        organization = words[0].strip()
    if organization is None and len(words) >= 2:
        inferred_org_tokens: list[str] = []
        for original, lowered_token in zip(words, tokens):
            if lowered_token in _ORG_STOPWORDS:
                break
            if lowered_token in _KNOWN_LOCATION_TOKENS:
                break
            if lowered_token in _PLATFORM_HINTS:
                break
            inferred_org_tokens.append(original)
        if inferred_org_tokens and any(char.isupper() for char in "".join(inferred_org_tokens)):
            organization = " ".join(inferred_org_tokens).strip()
    if organization is None:
        first_token_is_role = tokens[0] in _ROLE_HINTS if tokens else False
        if len(words) >= 2 and not first_token_is_role:
            trailing_location_idx = None
            for idx, token in enumerate(tokens):
                if token in _KNOWN_LOCATION_TOKENS:
                    trailing_location_idx = idx
                    break
            if trailing_location_idx and trailing_location_idx > 0:
                org_candidate = _clean_fragment(" ".join(words[:trailing_location_idx]))
                if org_candidate and not _looks_role_like(org_candidate):
                    organization = org_candidate
    if organization:
        organization = re.sub(r"\b(founder|cofounder|co-founder|ceo|executive)\b.*$", "", organization, flags=re.I).strip()
        organization = _clean_fragment(organization)
        if any(organization.casefold().startswith(prefix) for prefix in _ORG_VERB_PREFIXES):
            organization = ""
        organization_tokens = [token.casefold() for token in re.findall(r"[A-Za-zÀ-ÿ0-9]+", organization)]
        if not organization_tokens or all(token in _ORG_LINKING_STOPWORDS for token in organization_tokens) or _looks_role_like(organization):
            organization = None
    if organization and location and organization.casefold() == location.casefold():
        location = None

    activity_tokens = [token for token in tokens if token in _ACTIVITY_HINTS]
    if activity_tokens:
        domain_activity = " ".join(dict.fromkeys(activity_tokens))

    if role is None:
        if "football" in tokens:
            role = "footballer"
            if domain_activity is None:
                domain_activity = "football"
        elif "music" in tokens and "singer" in tokens:
            role = "singer"
            domain_activity = "music"

    generic_terms = tuple(
        token for token in dict.fromkeys(tokens) if token not in _STOP_TOKENS and token not in _LOCATION_PREPOSITIONS
    )
    if (
        domain_activity is None
        and organization is None
        and normalized
        and len(tokens) <= 4
        and not any(char.isdigit() for char in normalized)
        and role is None
    ):
        filtered = [word for word in words if word.casefold() not in _KNOWN_LOCATION_TOKENS and word.casefold() not in _PLATFORM_HINTS]
        if filtered:
            domain_activity = " ".join(filtered).strip()

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
        return [f"\"{base}\"", f"{base} linkedin", f"{base} profile", base]
    context_data = _parse_context(context)
    context_clean = context.strip() if context else ""
    if not context_clean:
        return [base, f"\"{base}\""]
    variants: list[str] = [f"{base} {context_clean}", f"\"{base}\" {context_clean}"]

    # Identity-first family.
    variants.extend([f"\"{base}\"", f"{base} linkedin", f"{base} profile", f"{base} perfil"])
    if context_data.organization:
        variants.append(f"\"{base}\" \"{context_data.organization}\"")
        variants.append(f"{base} {context_data.organization} linkedin")
    if context_data.location:
        variants.append(f"\"{base}\" \"{context_data.location}\"")

    # Name + role family.
    if context_data.role:
        variants.append(f"\"{base}\" \"{context_data.role}\"")
        variants.append(f"{base} {context_data.role} linkedin")
        variants.append(f"{base} {context_clean} linkedin")
        variants.append(f"{base} {context_clean} profile")

    # Name + org + role family.
    if context_data.organization and context_data.role:
        variants.extend(
            [
                f"\"{base}\" \"{context_data.organization}\" \"{context_data.role}\"",
                f"{base} {context_data.organization} {context_data.role} linkedin",
            ]
        )

    # Name + org + location family.
    if context_data.organization and context_data.location:
        variants.append(f"\"{base}\" \"{context_data.organization}\" \"{context_data.location}\"")

    # Name + role + location family.
    if context_data.role and context_data.location:
        variants.append(f"\"{base}\" \"{context_data.role}\" \"{context_data.location}\"")

    # Conservatively include raw context and lightweight local-language expansion.
    variants.append(f"\"{base}\" {context_clean}")
    if context_data.organization:
        variants.append(f"{base} {context_data.organization} perfil")
        variants.append(f"{base} {context_data.organization} empresa")
        variants.append(f"{base} {context_data.organization} equipo")

    if context_data.media_platform:
        variants.append(f"{base} {context_data.media_platform}")
    if context_data.institutional_hint:
        variants.append(f"{base} {context_data.institutional_hint} profile")

    # Name-only fallback family.
    if not (context_data.organization and context_data.location):
        variants.append(base)
    variants.append(f"{base} linkedin")

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
    enable_name_only_fallback: bool,
) -> tuple[list[Mapping[str, str]], tuple[str, ...]]:
    candidate_pool: list[tuple[str, str, str, str, str, str, str, Mapping[str, str]]] = []
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
        if enable_name_only_fallback and len(merged_sources) <= 1 and " " in query_text:
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
            raw_materialized = {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or item.get("link") or "").strip(),
                "snippet": str(item.get("snippet") or item.get("description") or "").strip(),
            }
            canonical_url = _canonicalize_url(raw_materialized["url"])
            materialized = {
                "title": " ".join(raw_materialized["title"].split()),
                "url": canonical_url or raw_materialized["url"],
                "snippet": " ".join(raw_materialized["snippet"].split()),
            }
            title_signature = " ".join(materialized["title"].casefold().split())
            snippet_signature = " ".join(materialized["snippet"].casefold().split())
            domain_signature = urlparse(canonical_url).netloc.casefold()
            candidate_pool.append(
                (
                    canonical_url,
                    title_signature,
                    snippet_signature,
                    domain_signature,
                    materialized["title"],
                    raw_materialized["url"],
                    materialized["snippet"],
                    materialized,
                )
            )

    candidate_pool.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6]))

    combined_results: list[Mapping[str, str]] = []
    seen_urls: set[str] = set()
    seen_signatures: set[tuple[str, str]] = set()
    for canonical_url, title_signature, _snippet_signature, domain_signature, *_keys, item in candidate_pool:
        signature = (title_signature, domain_signature)
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
    filtered_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    query_pairs = sorted(filtered_pairs, key=lambda item: (item[0], item[1]))
    netloc = parsed.netloc.casefold().removeprefix("www.")
    if netloc.endswith(":80") and parsed.scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and parsed.scheme == "https":
        netloc = netloc[:-4]
    normalized_path = re.sub(r"/+", "/", parsed.path.strip()) or "/"
    normalized_path = normalized_path if normalized_path == "/" else normalized_path.rstrip("/") or "/"
    cleaned = parsed._replace(
        netloc=netloc,
        query=urlencode(query_pairs, doseq=True),
        fragment="",
        path=normalized_path,
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
    execution = _execute_resolution_pipeline(
        target_name,
        context=context,
        html_search=html_search,
        instant_search=instant_search,
    )
    return execution.resolved, list(execution.ranked)


def resolve_query_with_debug(
    target_name: str,
    *,
    context: str | None,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> tuple[ResolutionOutput | None, list[ScoredCandidate], SearchPipelineDiagnostics]:
    execution = _execute_resolution_pipeline(
        target_name,
        context=context,
        html_search=html_search,
        instant_search=instant_search,
    )

    diagnostics = SearchPipelineDiagnostics(
        raw_results_count=len(execution.combined_results),
        normalized_results_count=execution.normalized_results_count,
        filtered_results_count=len(execution.search_results),
        ranked_candidates_count=len(execution.ranked),
        filter_decisions=execution.filter_decisions,
        ranked_candidates=execution.ranked,
        query_attempts=execution.query_attempts,
        source_diversity_count=len({item.result.domain for item in execution.ranked}),
        ambiguity_triggered=bool(execution.resolved and execution.resolved.ambiguity_detected),
        ambiguity_reason=execution.resolved.ambiguity_reason or "" if execution.resolved else "",
        context_interpretation=(
            f"role={execution.parsed_context.role or '-'}; org={execution.parsed_context.organization or '-'}; "
            f"location={execution.parsed_context.location or '-'}; activity={execution.parsed_context.domain_activity or '-'}; "
            f"platform={execution.parsed_context.media_platform or '-'}; institutional={execution.parsed_context.institutional_hint or '-'}; "
            f"generic={','.join(execution.parsed_context.generic_terms[:6]) if execution.parsed_context.generic_terms else '-'}"
        ),
        query_validity=execution.query_validity,
    )
    return execution.resolved, list(execution.ranked), diagnostics


def _execute_resolution_pipeline(
    target_name: str,
    *,
    context: str | None,
    html_search: Callable[[str], list[Mapping[str, str]]],
    instant_search: Callable[[str], list[Mapping[str, str]]],
) -> _PipelineExecution:
    parsed_context = _parse_context(context)
    name_only_fallback_enabled = not (parsed_context.organization or parsed_context.location)
    combined_results, query_attempts = _run_search(
        _query_variants(target_name, context),
        html_search=html_search,
        instant_search=instant_search,
        enable_name_only_fallback=name_only_fallback_enabled,
    )

    normalized_results = normalize_search_results(combined_results)
    search_results, filter_decisions = filter_search_results(normalized_results)
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

    if resolved is not None:
        ranked = _align_ranked_with_resolution(ranked, resolved)

    return _PipelineExecution(
        combined_results=tuple(combined_results),
        query_attempts=query_attempts,
        normalized_results_count=len(normalized_results),
        search_results=tuple(search_results),
        filter_decisions=tuple(filter_decisions),
        ranked=tuple(ranked),
        parsed_context=parsed_context,
        resolved=resolved,
        query_validity=assess_query_validity(target_name).status,
    )


def _align_ranked_with_resolution(ranked: list[ScoredCandidate], resolved: ResolutionOutput) -> list[ScoredCandidate]:
    if not ranked or not resolved.source_url:
        return ranked
    for idx, candidate in enumerate(ranked):
        if candidate.result.url == resolved.source_url:
            if idx == 0:
                return ranked
            return [candidate, *ranked[:idx], *ranked[idx + 1 :]]
    return ranked
