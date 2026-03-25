from __future__ import annotations

import re
from dataclasses import dataclass
import json
from html.parser import HTMLParser
from typing import Callable, Iterable, Literal
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .detection import is_directory_url
from .normalization import normalize_name
from .search import SearchResult

EntityType = Literal[
    "official_profile",
    "official_bio",
    "institutional_profile",
    "person_profile",
    "media_article",
    "reference_entry",
    "directory_listing",
    "aggregator_profile",
    "generic_article",
    "unknown",
]
NameMatchQuality = Literal["full_match", "reordered_match", "partial_match", "weak_match"]
SourceAuthorityTier = Literal[
    "official_institutional",
    "strong_encyclopedic",
    "reputable_media",
    "public_structured_profile",
    "directory_aggregator",
    "low_authority_bio_seo",
    "junk_or_malformed",
]

_DIRECTORY_HOST_HINTS = {"zoominfo", "rocketreach", "spokeo", "beenverified", "whitepages", "officialboard"}
_COMPANY_HINTS = {"about", "company", "team", "leadership", "careers", "executive", "management"}
_ARTICLE_HINTS = {"news", "blog", "article", "press"}
_EVENT_HINTS = {"event", "events", "conference", "keynote", "summit", "webinar", "session"}
_PROFILE_SEGMENTS = {"in", "u", "user", "profile", "people"}
_OFFICIAL_PROFILE_SEGMENTS = {
    "leadership",
    "executive",
    "management",
    "faculty",
    "staff",
    "team",
    "professionals",
    "attorneys",
    "lawyers",
    "doctors",
    "psychologists",
    "bio",
    "biography",
}
_CONTEXT_STOPWORDS = {"the", "a", "an", "at", "in", "de", "del", "la", "el", "of", "and"}
_HIGH_SIGNAL_DOMAINS = {"wikipedia.org", "microsoft.com", "google.com", "nvidia.com", "github.com"}
_NETWORK_PROFILE_DOMAINS = {"linkedin.com", "x.com", "twitter.com", "facebook.com", "instagram.com"}
_ACADEMIC_DOMAIN_HINTS = {".edu", ".ac.", ".edu."}
_MEDIA_DOMAIN_HINTS = {"imdb.com", "filmaffinity.com", "tmdb.org", "variety.com", "rollingstone.com"}
_CREATOR_PLATFORM_DOMAINS = {"youtube.com", "twitch.tv", "tiktok.com", "patreon.com", "soundcloud.com", "substack.com"}
_AGGREGATOR_HINTS = {"fandom", "wikidata", "wikia", "allfamous", "famousbirthdays"}
_REFERENCE_HINTS = {"wikipedia.org", "britannica.com", "wikidata.org"}
_OFFICIAL_DOMAIN_HINTS = {
    ".gov",
    ".edu",
    ".ac.",
    "un.org",
    "europa.eu",
    "who.int",
    "microsoft.com",
    "google.com",
    "nvidia.com",
    "apple.com",
    "openai.com",
    "vatican.va",
    "vaticannews.va",
    "messi.com",
}
_REPUTABLE_MEDIA_HINTS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
    "variety.com",
    "rollingstone.com",
}
_LOW_AUTHORITY_BIO_HINTS = {
    "biography",
    "bio",
    "networth",
    "age",
    "family",
    "relationship",
    "facts",
    "height",
    "wiki",
}
_NON_PERSON_TITLE_HINTS = {
    "news",
    "report",
    "policy",
    "privacy",
    "terms",
    "overview",
    "conference",
    "department",
    "company",
    "about us",
    "investor",
    "careers",
    "events",
    "announcement",
}
_ORG_PATH_HINTS = {
    "about",
    "company",
    "research",
    "department",
    "investors",
    "careers",
    "privacy",
    "terms",
    "newsroom",
}
_PERSONISH_TOKENS = {"mr", "mrs", "ms", "dr", "prof", "ceo", "founder", "president"}
_NOISE_TEXT_HINTS = {
    "privacy",
    "cookie",
    "consent",
    "terms",
    "your privacy choices",
    "opt out",
    "sign in",
    "subscribe",
    "menu",
    "navigation",
}
_PERSON_ROLE_HINTS = {
    "ceo",
    "chief",
    "founder",
    "president",
    "engineer",
    "director",
    "manager",
    "professor",
    "attorney",
    "lawyer",
    "scientist",
}
_NON_CORPORATE_OFFICIAL_HINTS = {
    "vatican.va",
    "vaticannews.va",
    "messi.com",
    "fifa.com",
    "uefa.com",
    "olympics.com",
}
_PRIMARY_OFFICIAL_ENTITY_TYPES = {"official_profile", "official_bio", "institutional_profile"}
_STRUCTURED_PROFILE_DOMAINS = {"researchgate.net", "orcid.org", "about.me", "academia.edu"}
_PLATFORM_SECONDARY_HINTS = {"gaming", "espanol", "español", "clips", "live", "shorts", "podcast", "music", "records"}
_DISTINCTIVE_TITLES = {"pope"}
_COMMON_NAME_TOKENS = {
    "john",
    "david",
    "carlos",
    "maria",
    "martinez",
    "perez",
    "lopez",
    "smith",
    "garcia",
}


@dataclass(slots=True, frozen=True)
class ContextQuery:
    """Optional context to disambiguate person identity resolution."""

    role: str | None = None
    organization: str | None = None
    location: str | None = None
    domain_activity: str | None = None
    media_platform: str | None = None
    institutional_hint: str | None = None
    raw_context: str | None = None
    generic_terms: tuple[str, ...] = ()
    expected_domains: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ScoredCandidate:
    """Scored candidate with deterministic evidence and debugging details."""

    result: SearchResult
    score: float
    entity_type: EntityType
    name_match: NameMatchQuality
    context_strength: float
    authority_tier: SourceAuthorityTier
    seo_penalty: float
    is_noise: bool
    reasons: tuple[str, ...]
    query_validity: str = "valid"
    score_cap_applied: float | None = None
    typing_confidence: float = 0.0


@dataclass(slots=True, frozen=True)
class TopCandidateContent:
    """Lightweight extraction from the top-ranked candidate page."""

    page_title: str
    og_title: str
    meta_description: str
    headings: tuple[str, ...]
    main_text: str
    json_ld_person_names: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ResolutionOutput:
    """Structured, evidence-driven output for a chosen identity candidate."""

    normalized_candidate_name: str
    source_url: str
    final_score: float
    entity_type: EntityType
    same_person_probability: float
    context_match_probability: float
    possible_role: str | None
    possible_organization: str | None
    possible_location: str | None
    explanation: str
    resolution_path: Literal["full_content", "partial_content", "search_only", "fetch_blocked"] = "search_only"
    fetch_status: str = "not_attempted"
    confidence_label: Literal["high", "medium", "low"] = "medium"
    ambiguity_detected: bool = False
    ambiguity_reason: str | None = None
    no_resolution: bool = False
    no_resolution_reason: str | None = None


@dataclass(slots=True, frozen=True)
class QueryValidityAssessment:
    status: Literal["valid", "too_short", "too_generic", "non_person_like", "numeric_or_garbage"]
    penalty: float
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class IdentityClusterEvidence:
    key: str
    candidates: tuple[ScoredCandidate, ...]
    representative: ScoredCandidate
    aggregate_score: float
    independent_domains: int
    official_support_count: int
    strong_source_count: int
    single_domain_only: bool
    creator_asset_hierarchy_score: float


def _normalized_tokens(value: str) -> list[str]:
    normalized = normalize_name(value)
    return [token for token in normalized.split(" ") if token]


def _joined_text(result: SearchResult) -> str:
    return " ".join(part for part in [result.title, result.snippet or "", result.url] if part).casefold()


def _domain_tokens(domain: str) -> set[str]:
    raw_parts = re.split(r"[.\-_/]+", domain.casefold())
    return {part for part in raw_parts if part and part not in {"www", "com", "org", "net"}}


def _context_overlap_score(context_value: str, haystack_tokens: set[str], *, label: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    tokens = [token for token in _normalized_tokens(context_value) if token not in _CONTEXT_STOPWORDS]
    if not tokens:
        return 0.0, reasons
    token_set = set(tokens)
    overlap = token_set & haystack_tokens
    coverage = len(overlap) / len(token_set)
    if coverage > 0:
        reasons.append(f"{label}_token_overlap:{coverage:.2f}")
    if coverage >= 0.99:
        reasons.append(f"{label}_full_coverage")
    return coverage, reasons


def detect_entity_type(result: SearchResult) -> EntityType:
    """Classify result into public identity source categories."""
    return detect_entity_type_with_reasons(result)[0]


def _looks_like_person_slug(slug: str) -> bool:
    parts = [part for part in re.split(r"[-_]", slug.casefold()) if part]
    alpha_parts = [part for part in parts if any(char.isalpha() for char in part)]
    if len(alpha_parts) < 2:
        return False
    return not any(part in _ARTICLE_HINTS | _ORG_PATH_HINTS for part in alpha_parts)


def _looks_like_person_title(title: str) -> bool:
    lowered = title.casefold()
    if any(hint in lowered for hint in _NON_PERSON_TITLE_HINTS):
        return False
    tokens = [token for token in _normalized_tokens(title) if token]
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    if len(alpha_tokens) >= 2:
        return True
    return any(token in _PERSONISH_TOKENS for token in alpha_tokens)


def detect_entity_type_with_reasons(result: SearchResult) -> tuple[EntityType, tuple[str, ...]]:
    """Classify result and emit deterministic rationale for debug mode."""

    parsed = urlparse(result.url)
    domain = parsed.netloc.casefold().removeprefix("www.") or result.domain.casefold()
    path_segments = [segment.casefold() for segment in parsed.path.split("/") if segment]

    joined_text = _joined_text(result)
    tail_segment = path_segments[-1] if path_segments else ""

    reasons: list[str] = []
    personish_tail = _looks_like_person_slug(tail_segment)
    personish_title = _looks_like_person_title(result.title)

    if is_directory_url(result.url) or any(token in domain for token in _DIRECTORY_HOST_HINTS):
        return "directory_listing", ("directory_pattern",)
    if any(token in domain for token in _AGGREGATOR_HINTS):
        return "aggregator_profile", ("aggregator_domain_pattern",)
    if any(domain.endswith(token) for token in _REFERENCE_HINTS):
        return "reference_entry", ("reference_domain_pattern",)

    if any(segment in _EVENT_HINTS for segment in path_segments):
        return "media_article", ("event_or_keynote_path_pattern",)
    if any(segment in _ARTICLE_HINTS for segment in path_segments) or any(hint in joined_text for hint in (" breaking ", "headline", "op-ed")):
        return "media_article", ("article_path_or_title_pattern",)

    if any(domain.endswith(platform_domain) for platform_domain in _CREATOR_PLATFORM_DOMAINS | _NETWORK_PROFILE_DOMAINS):
        if any(segment in path_segments for segment in {"channel", "c", "user", "creator", "in", "u", "profile"}) or any(
            segment.startswith("@") for segment in path_segments
        ):
            return "person_profile", ("public_profile_platform_path",)
        return "generic_article", ("platform_non_profile_page",)

    if any(domain.endswith(media_domain) for media_domain in _MEDIA_DOMAIN_HINTS):
        return "media_article", ("media_domain",)

    if len(path_segments) >= 2 and path_segments[-2] in _PROFILE_SEGMENTS:
        if personish_tail and personish_title:
            return "person_profile", ("profile_segment_with_personish_slug_and_title",)
        return "unknown", ("profile_segment_without_person_plausibility",)

    if any(hint in domain for hint in _ACADEMIC_DOMAIN_HINTS) or any(
        segment in path_segments for segment in {"faculty", "research", "department", "academics", "professor"}
    ):
        if any(segment in path_segments for segment in {"faculty", "professor", "staff", "people"}):
            return "institutional_profile", ("academic_domain_or_path",)
        return "institutional_profile", ("institutional_academic_path",)

    if len(path_segments) >= 2 and path_segments[-2] in _OFFICIAL_PROFILE_SEGMENTS:
        if personish_tail and (re.search(r"[-_]", tail_segment) or personish_title):
            return "official_profile", ("official_segment_with_personish_slug",)
        if any(token in joined_text for token in ("professor", "attorney", "psychologist", "md", "phd", "chief", "ceo")):
            return "official_profile", ("official_segment_with_role_tokens",)
        return "institutional_profile", ("official_segment_without_person_signals",)

    if domain in _NON_CORPORATE_OFFICIAL_HINTS:
        if personish_tail or personish_title:
            return "official_bio", ("non_corporate_official_domain_person_signal",)
        return "institutional_profile", ("non_corporate_official_domain",)

    if any(segment in _COMPANY_HINTS for segment in path_segments):
        if personish_tail or personish_title:
            return "official_bio", ("company_path_with_person_signals",)
        return "institutional_profile", ("company_path_without_person_signals",)
    if personish_tail and any(domain.endswith(suffix) for suffix in _OFFICIAL_DOMAIN_HINTS):
        return "official_bio", ("official_domain_person_slug",)
    if any(segment in _ORG_PATH_HINTS for segment in path_segments):
        return "institutional_profile", ("organization_path_pattern",)

    if re.search(r"\b(profile|bio|about\s+me|executive\s+profile|faculty|attorney|psychologist)\b", joined_text):
        if personish_title:
            return "person_profile", ("profile_or_bio_text_with_person_title",)
        reasons.append("profile_text_without_person_title")

    return "unknown", tuple(reasons or ["no_strong_entity_pattern"])


def detect_name_match_quality(query_name: str, result: SearchResult) -> NameMatchQuality:
    """Score name alignment using title+snippet+url evidence."""

    query_tokens = _normalized_tokens(query_name)
    haystack_tokens = _normalized_tokens(_joined_text(result))

    if not query_tokens:
        return "weak_match"

    query_norm = " ".join(query_tokens)
    haystack_norm = " ".join(haystack_tokens)
    title_tokens = _normalized_tokens(result.title)
    snippet_tokens = _normalized_tokens(result.snippet or "")

    if query_norm and query_norm in haystack_norm:
        return "full_match"

    if len(query_tokens) >= 2 and (
        set(query_tokens).issubset(set(title_tokens)) or set(query_tokens).issubset(set(snippet_tokens))
    ):
        return "reordered_match"

    overlap = len(set(query_tokens) & set(haystack_tokens))
    ratio = overlap / len(set(query_tokens))

    if ratio >= 0.5:
        return "partial_match"
    return "weak_match"


def context_match_strength(result: SearchResult, context: ContextQuery) -> tuple[float, tuple[str, ...]]:
    """Compute context match score using flexible public-identity hints."""

    reasons: list[str] = []
    score = 0.0
    haystack = _joined_text(result)
    haystack_tokens = set(_normalized_tokens(haystack))
    haystack_tokens.update(_domain_tokens(result.domain))

    def _score_hint(value: str | None, label: str, phrase_weight: float, overlap_weight: float) -> None:
        nonlocal score
        if not value:
            return
        normalized_value = normalize_name(value)
        if not normalized_value:
            return
        if normalized_value in haystack:
            score += phrase_weight
            reasons.append(f"{label}_phrase_match")
        overlap_score, overlap_reasons = _context_overlap_score(normalized_value, haystack_tokens, label=label)
        score += overlap_weight * overlap_score
        reasons.extend(overlap_reasons)

    if context.role:
        role = normalize_name(context.role)
        if role:
            if role in haystack:
                score += 0.38
                reasons.append("role_phrase_match")
            role_tokens = [token for token in _normalized_tokens(role) if token not in _CONTEXT_STOPWORDS]
            if len(role_tokens) >= 2 and all(token in haystack_tokens for token in role_tokens):
                score += 0.12
                reasons.append("role_reordered_match")
            overlap_score, overlap_reasons = _context_overlap_score(role, haystack_tokens, label="role")
            score += 0.28 * overlap_score
            reasons.extend(overlap_reasons)

    _score_hint(context.organization, "organization", 0.3, 0.2)
    _score_hint(context.location, "location", 0.3, 0.25)
    _score_hint(context.domain_activity, "activity", 0.2, 0.15)
    _score_hint(context.media_platform, "platform", 0.2, 0.15)
    _score_hint(context.institutional_hint, "institutional", 0.2, 0.14)

    if context.generic_terms:
        matched_terms = [term for term in context.generic_terms if term in haystack_tokens]
        if matched_terms:
            score += min(0.18, 0.06 * len(matched_terms))
            reasons.append(f"generic_terms_match:{','.join(sorted(matched_terms)[:4])}")

    if context.expected_domains and any(result.domain.endswith(domain) for domain in context.expected_domains):
        score += 0.2
        reasons.append("domain_relevance")

    if context.role and context.location:
        if all(token in haystack_tokens for token in _normalized_tokens(context.role)[:2]) and any(
            token in haystack_tokens for token in _normalized_tokens(context.location)
        ):
            score += 0.1
            reasons.append("combined_context_alignment")

    return min(score, 1.0), tuple(reasons)


def is_noise_result(result: SearchResult) -> bool:
    """Detect low-signal pages such as directories, list pages, and SEO aggregators."""

    text = _joined_text(result)
    noisy_phrases = ("top ", "best ", "list of", "find people", "people search")

    return (
        detect_entity_type(result) in {"directory_listing", "aggregator_profile"}
        or any(phrase in text for phrase in noisy_phrases)
        or re.search(r"\b(aggregator|directory|listing)\b", text) is not None
    )


def assess_query_validity(query_name: str) -> QueryValidityAssessment:
    normalized = normalize_name(query_name)
    tokens = [token for token in normalized.split() if token]
    compact = re.sub(r"\s+", "", query_name)
    reasons: list[str] = []

    if not tokens or len(compact) <= 1:
        return QueryValidityAssessment("too_short", 0.55, ("query_too_short",))
    if re.fullmatch(r"[\d\W_]+", compact) or re.fullmatch(r"\d{4,}", compact):
        return QueryValidityAssessment("numeric_or_garbage", 0.65, ("numeric_or_garbage_query",))
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return QueryValidityAssessment("too_short", 0.45, ("single_token_too_short",))
    if len(tokens) == 1 and tokens[0] in {"person", "name", "profile", "biography"}:
        return QueryValidityAssessment("too_generic", 0.45, ("generic_single_token_query",))
    if sum(any(char.isalpha() for char in token) for token in tokens) == 0:
        return QueryValidityAssessment("non_person_like", 0.5, ("no_alpha_person_signal",))
    if len(tokens) > 6:
        reasons.append("long_freeform_query")
    return QueryValidityAssessment("valid", 0.0, tuple(reasons))


def detect_source_authority(result: SearchResult) -> tuple[SourceAuthorityTier, float, list[str]]:
    domain = result.domain.casefold()
    url = result.url.casefold()
    title = result.title.casefold()
    snippet = (result.snippet or "").casefold()
    text = f"{title} {snippet} {url}"
    reasons: list[str] = []

    if not domain or domain.count(".") == 0:
        return "junk_or_malformed", -0.35, ["malformed_domain"]

    if domain.endswith("wikipedia.org") or "britannica.com" in domain:
        return "strong_encyclopedic", 0.18, ["encyclopedic_source"]
    if any(domain.endswith(suffix) for suffix in _OFFICIAL_DOMAIN_HINTS):
        return "official_institutional", 0.26, ["official_or_institutional_domain"]
    if any(domain.endswith(hint) for hint in _NON_CORPORATE_OFFICIAL_HINTS):
        return "official_institutional", 0.25, ["non_corporate_official_domain"]
    if any(domain.endswith(media) for media in _REPUTABLE_MEDIA_HINTS):
        return "reputable_media", 0.16, ["reputable_media_domain"]
    if any(domain.endswith(network) for network in _NETWORK_PROFILE_DOMAINS):
        return "public_structured_profile", 0.08, ["structured_public_profile"]
    if detect_entity_type(result) in {"directory_listing", "aggregator_profile"}:
        return "directory_aggregator", -0.26, ["directory_or_aggregator_source"]
    if any(hint in text for hint in _LOW_AUTHORITY_BIO_HINTS):
        return "low_authority_bio_seo", -0.24, ["low_authority_biography_patterns"]
    return "public_structured_profile", 0.02, ["default_public_profile_assumption"]


def _seo_bio_penalty(result: SearchResult, authority_tier: SourceAuthorityTier) -> tuple[float, list[str]]:
    text = _joined_text(result)
    penalty = 0.0
    reasons: list[str] = []
    pattern_hits = [
        "net worth",
        "age",
        "family",
        "relationship",
        "biography",
        "facts",
        "career",
        "husband",
        "wife",
    ]
    hits = [term for term in pattern_hits if term in text]
    if len(hits) >= 2:
        penalty += 0.14
        reasons.append(f"seo_bio_terms:{','.join(hits[:4])}")
    if re.search(r"\b(age|net worth|family|height)\b.*\b(age|net worth|family|height)\b", text):
        penalty += 0.08
        reasons.append("repetitive_personal_info_phrasing")
    if detect_entity_type(result) in {"media_article", "generic_article", "unknown"} and "biography" in text and result.domain.count(".") >= 1:
        penalty += 0.06
        reasons.append("generic_biography_page_penalty")
    if authority_tier in {"strong_encyclopedic", "official_institutional"}:
        adjusted = penalty * 0.2
        if penalty > 0:
            reasons.append("authority_conditioned_seo_penalty_reduction")
        penalty = adjusted
    return min(0.3, penalty), reasons


def score_candidate(
    result: SearchResult,
    query_name: str,
    context: ContextQuery,
    query_validity: QueryValidityAssessment,
) -> ScoredCandidate:
    """Deterministic scoring with explicit bonuses/penalties for explainability."""

    reasons: list[str] = []
    entity_type, entity_rationale = detect_entity_type_with_reasons(result)
    name_match = detect_name_match_quality(query_name, result)
    context_strength, context_reasons = context_match_strength(result, context)
    authority_tier, authority_weight, authority_reasons = detect_source_authority(result)
    seo_penalty, seo_reasons = _seo_bio_penalty(result, authority_tier)
    noisy = is_noise_result(result)

    score = 0.0

    entity_weights: dict[EntityType, float] = {
        "official_profile": 0.4,
        "official_bio": 0.34,
        "institutional_profile": 0.24,
        "person_profile": 0.3,
        "media_article": 0.08,
        "reference_entry": 0.14,
        "directory_listing": -0.28,
        "aggregator_profile": -0.24,
        "generic_article": -0.08,
        "unknown": 0.0,
    }
    score += entity_weights[entity_type]
    reasons.append(f"entity:{entity_type}")
    reasons.extend(f"entity_rationale:{signal}" for signal in entity_rationale[:3])

    name_weights: dict[NameMatchQuality, float] = {
        "full_match": 0.34,
        "reordered_match": 0.26,
        "partial_match": 0.14,
        "weak_match": 0.03,
    }
    score += name_weights[name_match]
    reasons.append(f"name:{name_match}")
    score += authority_weight
    reasons.append(f"authority:{authority_tier}")
    reasons.extend(authority_reasons)

    snippet_bonus = 0.0
    if result.snippet:
        snippet_terms = set(_normalized_tokens(result.snippet))
        query_terms = set(_normalized_tokens(query_name))
        if query_terms:
            snippet_bonus = 0.15 * (len(snippet_terms & query_terms) / len(query_terms))
            score += snippet_bonus
    reasons.append(f"snippet_bonus:{snippet_bonus:.2f}")

    score += 0.5 * context_strength
    reasons.extend(context_reasons)

    has_context_constraints = any(
        [
            context.role,
            context.organization,
            context.location,
            context.domain_activity,
            context.media_platform,
            context.institutional_hint,
            context.generic_terms,
            context.expected_domains,
        ]
    )
    if has_context_constraints and name_match in {"full_match", "reordered_match"} and context_strength < 0.12:
        score -= 0.3
        reasons.append("weak_context_exact_name_penalty")

    if entity_type in {"person_profile", "official_profile", "official_bio", "institutional_profile"}:
        score += 0.08
        reasons.append("profile_structure_bonus")

    exact_name = name_match in {"full_match", "reordered_match"}
    explicit_org_alignment = bool(context.organization and normalize_name(context.organization) in _joined_text(result))
    if context.role and entity_type in {"official_profile", "official_bio"}:
        score += 0.06
        reasons.append("role_official_alignment_bonus")
    if (
        entity_type in _PRIMARY_OFFICIAL_ENTITY_TYPES
        and authority_tier == "official_institutional"
        and exact_name
        and explicit_org_alignment
    ):
        score += 0.16
        reasons.append("first_party_exact_name_org_priority")
    if context.media_platform and entity_type in {"person_profile", "media_article"}:
        score += 0.08
        reasons.append("creator_media_alignment_bonus")
    if context.institutional_hint and entity_type in {"institutional_profile", "official_bio"}:
        score += 0.08
        reasons.append("institutional_alignment_bonus")

    if any(result.domain.endswith(domain) for domain in _HIGH_SIGNAL_DOMAINS):
        score += 0.08
        reasons.append("high_signal_domain_bonus")

    if any(result.domain.endswith(domain) for domain in _NETWORK_PROFILE_DOMAINS):
        if has_context_constraints and context_strength < 0.35:
            score -= 0.18
            reasons.append("network_profile_low_context_penalty")
        else:
            score += 0.02
            reasons.append("network_profile_presence")

    if entity_type in {"media_article", "generic_article"} and has_context_constraints and context_strength < 0.22:
        score -= 0.12
        reasons.append("article_low_context_penalty")
    if entity_type in {"media_article", "generic_article"} and authority_tier not in {"strong_encyclopedic", "official_institutional"}:
        score -= 0.08
        reasons.append("generic_article_source_priority_penalty")
    if (
        has_context_constraints
        and entity_type in {"official_profile", "official_bio", "institutional_profile"}
        and authority_tier in {"official_institutional", "strong_encyclopedic", "public_structured_profile"}
    ):
        score += 0.08
        reasons.append("source_priority_boost:structured_or_official")

    if noisy:
        score -= 0.35
        reasons.append("noise_penalty")
    if seo_penalty > 0:
        score -= seo_penalty
        reasons.append(f"seo_penalty:{seo_penalty:.2f}")
        reasons.extend(seo_reasons)
    if result.domain.endswith("wikipedia.org") and context_strength < 0.15:
        score -= 0.08
        reasons.append("wikipedia_weak_context_penalty")
    score_cap_applied: float | None = None
    query_score_caps = {
        "too_short": 0.42,
        "numeric_or_garbage": 0.28,
        "too_generic": 0.48,
        "non_person_like": 0.4,
    }
    if query_validity.status in query_score_caps:
        score_cap_applied = query_score_caps[query_validity.status]
        score = min(score, score_cap_applied)
        reasons.append(f"query_invalid_score_cap:{query_validity.status}:{score_cap_applied:.2f}")

    return ScoredCandidate(
        result=result,
        score=max(0.0, min(1.0, score)),
        entity_type=entity_type,
        name_match=name_match,
        context_strength=context_strength,
        authority_tier=authority_tier,
        seo_penalty=seo_penalty,
        is_noise=noisy,
        query_validity=query_validity.status,
        score_cap_applied=score_cap_applied,
        typing_confidence=_entity_typing_confidence(entity_type, entity_rationale),
        reasons=tuple(reasons),
    )


def _platform_asset_owner(result: SearchResult) -> str:
    parsed = urlparse(result.url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    for segment in segments:
        lowered = segment.casefold()
        if lowered.startswith("@") and len(lowered) > 1:
            return normalize_name(lowered.removeprefix("@"))
    if len(segments) >= 2 and segments[0].casefold() in {"channel", "user", "c"}:
        return normalize_name(segments[1])
    title_owner = _clean_title_fragment(result.title).split("|")[0].strip()
    return normalize_name(title_owner)


def _platform_asset_priority(result: SearchResult) -> float:
    parsed = urlparse(result.url)
    segments = [segment.casefold() for segment in parsed.path.split("/") if segment]
    text = _joined_text(result)
    if any(segment == "watch" for segment in segments):
        return 0.2
    if any(token in text for token in _PLATFORM_SECONDARY_HINTS):
        return 0.55
    if any(segment in {"music", "videos", "shorts"} for segment in segments):
        return 0.45
    if any(segment.startswith("@") for segment in segments):
        return 1.0
    if len(segments) >= 2 and segments[0] in {"channel", "user", "c"}:
        return 0.9
    return 0.6


def _candidate_cluster_key(candidate: ScoredCandidate, query_name: str, context: ContextQuery) -> str:
    haystack = _joined_text(candidate.result)
    query_norm = normalize_name(query_name)
    query_tokens = [token for token in query_norm.split() if token]
    all_name_tokens_present = bool(query_tokens) and all(token in haystack for token in query_tokens)
    if candidate.result.domain.endswith(tuple(_CREATOR_PLATFORM_DOMAINS)):
        owner = _platform_asset_owner(candidate.result)
        if owner and query_norm and query_norm in owner:
            return f"platform_brand:{candidate.result.domain}:{query_norm}"
        if owner:
            return f"platform:{candidate.result.domain}:{owner}"
    if all_name_tokens_present and context.organization:
        org_norm = normalize_name(context.organization)
        if org_norm and org_norm in haystack:
            return f"org:{org_norm}|name:{query_norm}"
    if all_name_tokens_present and context.institutional_hint:
        institutional_norm = normalize_name(context.institutional_hint)
        if institutional_norm and institutional_norm in haystack:
            return f"institutional:{institutional_norm}|name:{query_norm}"
    if (
        all_name_tokens_present
        and candidate.authority_tier == "official_institutional"
        and candidate.entity_type in _PRIMARY_OFFICIAL_ENTITY_TYPES
    ):
        return f"official:{query_norm}"
    return f"url:{candidate.result.url.casefold()}"


def _cluster_identity_evidence(ranked: list[ScoredCandidate], query_name: str, context: ContextQuery) -> list[IdentityClusterEvidence]:
    grouped: dict[str, list[ScoredCandidate]] = {}
    for candidate in ranked:
        key = _candidate_cluster_key(candidate, query_name, context)
        grouped.setdefault(key, []).append(candidate)
    clusters: list[IdentityClusterEvidence] = []
    for key, members in grouped.items():
        representative = max(members, key=lambda item: _representative_priority(item, context))
        domains = {item.result.domain for item in members}
        official_support_count = sum(1 for item in members if item.authority_tier == "official_institutional")
        strong_source_count = sum(
            1
            for item in members
            if item.authority_tier in {"official_institutional", "reputable_media", "strong_encyclopedic"}
        )
        aggregate = representative.score
        aggregate += min(0.24, 0.08 * (len(members) - 1))
        aggregate += min(0.2, 0.06 * (len(domains) - 1))
        aggregate += min(0.16, 0.04 * strong_source_count)
        creator_asset_score = 0.0
        if representative.result.domain.endswith(tuple(_CREATOR_PLATFORM_DOMAINS)):
            creator_asset_score = max(_platform_asset_priority(item.result) for item in members)
            aggregate += 0.08 * creator_asset_score
        clusters.append(
            IdentityClusterEvidence(
                key=key,
                candidates=tuple(members),
                representative=representative,
                aggregate_score=min(1.0, aggregate),
                independent_domains=len(domains),
                official_support_count=official_support_count,
                strong_source_count=strong_source_count,
                single_domain_only=len(domains) <= 1,
                creator_asset_hierarchy_score=creator_asset_score,
            )
        )
    clusters.sort(
        key=lambda item: (
            -item.aggregate_score,
            -item.official_support_count,
            -item.independent_domains,
            -item.strong_source_count,
            -item.creator_asset_hierarchy_score,
            -item.representative.score,
        )
    )
    return clusters


def _is_person_specific_page(candidate: ScoredCandidate) -> bool:
    parsed = urlparse(candidate.result.url)
    segments = [segment.casefold() for segment in parsed.path.split("/") if segment]
    if candidate.entity_type in {"official_profile", "official_bio", "person_profile"}:
        return True
    if candidate.entity_type == "institutional_profile" and any(segment in {"faculty", "staff", "people", "leadership", "team"} for segment in segments):
        return True
    return False


def _representative_priority(candidate: ScoredCandidate, context: ContextQuery) -> tuple[float, ...]:
    parsed = urlparse(candidate.result.url)
    segments = [segment.casefold() for segment in parsed.path.split("/") if segment]
    official_person = (
        candidate.authority_tier == "official_institutional"
        and candidate.entity_type in {"official_bio", "official_profile", "institutional_profile"}
        and _is_person_specific_page(candidate)
    )
    official_event_or_news = any(segment in _EVENT_HINTS | _ARTICLE_HINTS for segment in segments)
    authority_priority = {
        "official_institutional": 5,
        "public_structured_profile": 4,
        "strong_encyclopedic": 3,
        "reputable_media": 2,
        "low_authority_bio_seo": 1,
        "directory_aggregator": 0,
        "junk_or_malformed": -1,
    }[candidate.authority_tier]
    entity_priority = {
        "official_bio": 6,
        "official_profile": 6,
        "institutional_profile": 5,
        "person_profile": 4,
        "reference_entry": 3,
        "media_article": 2,
        "generic_article": 1,
        "unknown": 0,
        "aggregator_profile": -1,
        "directory_listing": -2,
    }[candidate.entity_type]
    context_bonus = candidate.context_strength if context.raw_context else 0.0
    event_penalty = 0.7 if official_event_or_news else 0.0
    person_bonus = 0.6 if official_person else (0.2 if _is_person_specific_page(candidate) else 0.0)
    return (
        person_bonus + entity_priority + authority_priority - event_penalty,
        candidate.score + context_bonus,
        candidate.typing_confidence,
    )


def _query_distinctiveness(query_name: str) -> float:
    tokens = _normalized_tokens(query_name)
    if not tokens:
        return 0.0
    score = 0.0
    if len(tokens) >= 2:
        score += 0.28
    if len(tokens) >= 3:
        score += 0.14
    if any(token in _DISTINCTIVE_TITLES for token in tokens):
        score += 0.3
    if any(len(token) >= 7 for token in tokens):
        score += 0.12
    if len(tokens) == 1 and any(char.isalpha() for char in tokens[0]) and any(char.isupper() for char in query_name):
        score += 0.18
    common_hits = sum(1 for token in tokens if token in _COMMON_NAME_TOKENS)
    score -= 0.18 * common_hits
    return max(0.0, min(0.9, score))


def _entity_typing_confidence(entity_type: EntityType, reasons: tuple[str, ...]) -> float:
    if entity_type == "unknown":
        return 0.2
    strong_markers = ("official_segment", "reference_domain_pattern", "academic_domain_or_path", "public_profile_platform_path")
    if any(any(marker in reason for marker in strong_markers) for reason in reasons):
        return 0.9
    if len(reasons) >= 2:
        return 0.75
    return 0.6


class _HTMLSignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_heading: str | None = None
        self._in_ignored = False
        self._blocked_stack: list[bool] = []
        self.title = ""
        self.meta_description = ""
        self.og_title = ""
        self.headings: list[str] = []
        self._text: list[str] = []
        self._in_json_ld = False
        self._json_ld_chunks: list[str] = []
        self.json_ld_person_names: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        attrs_dict = {key.casefold(): (value or "") for key, value in attrs}
        class_id_blob = " ".join((attrs_dict.get("class", ""), attrs_dict.get("id", ""))).casefold()
        should_block = lowered in {"header", "footer", "nav", "aside", "svg", "button", "form"} or any(
            hint in class_id_blob for hint in {"cookie", "consent", "privacy", "footer", "header", "menu", "nav"}
        )
        self._blocked_stack.append(should_block)
        if lowered in {"script", "style", "noscript"}:
            self._in_ignored = True
        if lowered == "title":
            self._in_title = True
        if lowered in {"h1", "h2"}:
            self._in_heading = lowered
        if lowered == "meta" and attrs_dict.get("name", "").casefold() == "description":
            self.meta_description = attrs_dict.get("content", "").strip()
        if lowered == "meta" and attrs_dict.get("property", "").casefold() == "og:title":
            self.og_title = attrs_dict.get("content", "").strip()
        if lowered == "script" and "ld+json" in attrs_dict.get("type", "").casefold():
            self._in_json_ld = True

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if self._blocked_stack:
            self._blocked_stack.pop()
        if lowered in {"script", "style", "noscript"}:
            self._in_ignored = False
        if lowered == "script" and self._in_json_ld:
            self._in_json_ld = False
            payload = " ".join(self._json_ld_chunks).strip()
            self._json_ld_chunks.clear()
            if payload:
                self._extract_json_ld_person_names(payload)
        if lowered == "title":
            self._in_title = False
        if lowered in {"h1", "h2"}:
            self._in_heading = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._in_ignored or any(self._blocked_stack):
            return
        if self._in_json_ld:
            self._json_ld_chunks.append(data)
            return
        lowered = text.casefold()
        if any(hint in lowered for hint in _NOISE_TEXT_HINTS):
            return
        if len(text) > 220:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif self._in_heading:
            self.headings.append(text)
        else:
            self._text.append(text)

    @property
    def main_text(self) -> str:
        return " ".join(self._text)

    def _extract_json_ld_person_names(self, payload: str) -> None:
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return

        def _walk(node: object) -> None:
            if isinstance(node, dict):
                node_type = str(node.get("@type", "")).casefold()
                name = node.get("name")
                if "person" in node_type and isinstance(name, str) and name.strip():
                    self.json_ld_person_names.append(name.strip())
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(decoded)


def _safe_fetch_html(url: str, timeout: int = 8) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read(450_000)
    return body.decode("utf-8", errors="ignore")


def extract_top_candidate_content(
    url: str,
    *,
    fetcher: Callable[[str], str] | None = None,
) -> tuple[TopCandidateContent, str]:
    """Fetch and parse only basic visible content from top candidate."""

    try:
        html = (fetcher or _safe_fetch_html)(url)
    except HTTPError as exc:
        status = f"http_error:{exc.code}"
        return TopCandidateContent(page_title="", og_title="", meta_description="", headings=(), main_text=""), status
    except URLError:
        return TopCandidateContent(page_title="", og_title="", meta_description="", headings=(), main_text=""), "network_error"
    except TimeoutError:
        return TopCandidateContent(page_title="", og_title="", meta_description="", headings=(), main_text=""), "timeout"
    except Exception:
        return TopCandidateContent(page_title="", og_title="", meta_description="", headings=(), main_text=""), "fetch_error"

    if not html:
        return TopCandidateContent(page_title="", og_title="", meta_description="", headings=(), main_text=""), "empty_response"

    parser = _HTMLSignalParser()
    parser.feed(html)

    return TopCandidateContent(
        page_title=parser.title.strip(),
        og_title=parser.og_title.strip(),
        meta_description=parser.meta_description.strip(),
        headings=tuple(parser.headings[:8]),
        main_text=parser.main_text[:2500].strip(),
        json_ld_person_names=tuple(parser.json_ld_person_names[:3]),
    ), "ok"


def infer_profile_signals(content: TopCandidateContent) -> tuple[str, str | None, str | None, str | None]:
    """Infer normalized name, profession, organization, and location from page content."""

    merged = " | ".join(
        part for part in [content.page_title, content.meta_description, " ".join(content.headings), content.main_text] if part
    )
    normalized_name, _, _ = _derive_candidate_name(content)

    role_match = re.search(
        r"\b(engineer|developer|manager|director|analyst|researcher|journalist|designer|actor|actress|musician|professor|creator|streamer|author|lawyer|psychologist)\b",
        merged,
        re.I,
    )
    org_match = re.search(r"\b(?:at|with)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", merged)
    location_match = re.search(r"\b(?:based in|located in|from)\s+([A-Z][A-Za-z .-]{2,40})", merged)

    return (
        normalized_name,
        role_match.group(1).casefold() if role_match else None,
        org_match.group(1).strip() if org_match else None,
        location_match.group(1).strip() if location_match else None,
    )


def _clean_title_fragment(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" -|•:")
    cleaned = re.split(r"\s[|\-–:]\s", cleaned)[0].strip()
    return cleaned


def _looks_person_like_text(value: str) -> bool:
    if not value:
        return False
    lowered = value.casefold()
    if any(hint in lowered for hint in _NOISE_TEXT_HINTS):
        return False
    tokens = [token for token in _normalized_tokens(value) if token]
    if len(tokens) < 2:
        return False
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    if len(alpha_tokens) < 2:
        return False
    if any(token in _PERSON_ROLE_HINTS for token in alpha_tokens):
        return True
    return all(token not in _NON_PERSON_TITLE_HINTS for token in alpha_tokens[:3])


def _derive_candidate_name(content: TopCandidateContent) -> tuple[str, float, str]:
    sources: list[tuple[str, float, str]] = []
    if content.json_ld_person_names:
        sources.append((content.json_ld_person_names[0], 0.98, "json_ld_person_name"))
    if content.og_title:
        sources.append((_clean_title_fragment(content.og_title), 0.9, "og_title"))
    sources.extend([(_clean_title_fragment(h), 0.76, "h1_h2_heading") for h in content.headings[:2]])
    if content.page_title:
        sources.append((_clean_title_fragment(content.page_title), 0.68, "html_title"))
    if content.meta_description:
        sources.append((_clean_title_fragment(content.meta_description), 0.56, "meta_description"))
    for candidate, quality, source in sources:
        if _looks_person_like_text(candidate):
            return normalize_name(candidate), quality, source
    return "", 0.1, "weak_fallback"


def rank_candidates(results: Iterable[SearchResult], query_name: str, context: ContextQuery) -> list[ScoredCandidate]:
    """Rank all candidates deterministically using evidence-driven heuristics."""

    query_validity = assess_query_validity(query_name)
    scored = [score_candidate(result, query_name, context, query_validity) for result in results]
    authority_rank = {
        "official_institutional": 6,
        "public_structured_profile": 5,
        "reputable_media": 4,
        "strong_encyclopedic": 3,
        "low_authority_bio_seo": 2,
        "directory_aggregator": 1,
        "junk_or_malformed": 0,
    }
    return sorted(
        scored,
        key=lambda item: (
            -item.score,
            -item.context_strength,
            -authority_rank[item.authority_tier],
            -item.typing_confidence,
            item.seo_penalty,
            item.result.domain,
            item.result.url,
        ),
    )


def resolve_identity(
    query_name: str,
    results: Iterable[SearchResult],
    *,
    role: str | None = None,
    organization: str | None = None,
    location: str | None = None,
    domain_activity: str | None = None,
    media_platform: str | None = None,
    institutional_hint: str | None = None,
    raw_context: str | None = None,
    generic_terms: tuple[str, ...] = (),
    expected_domains: tuple[str, ...] = (),
    fetcher: Callable[[str], str] | None = None,
) -> ResolutionOutput | None:
    """Resolve the highest-confidence identity candidate and extract top-page signals."""

    context = ContextQuery(
        role=role,
        organization=organization,
        location=location,
        domain_activity=domain_activity,
        media_platform=media_platform,
        institutional_hint=institutional_hint,
        raw_context=raw_context,
        generic_terms=generic_terms,
        expected_domains=expected_domains,
    )
    ranked = rank_candidates(results, query_name, context)
    if not ranked:
        return None
    query_validity = assess_query_validity(query_name)

    invalid_hard_reject = query_validity.status in {"too_short", "numeric_or_garbage", "non_person_like"}
    if invalid_hard_reject:
        return ResolutionOutput(
            normalized_candidate_name="",
            source_url="",
            final_score=0.0,
            entity_type="unknown",
            same_person_probability=0.0,
            context_match_probability=0.0,
            possible_role=None,
            possible_organization=None,
            possible_location=None,
            explanation=f"NO_RESOLUTION: invalid query ({query_validity.status}); hard rejection triggered.",
            resolution_path="search_only",
            fetch_status="not_attempted",
            confidence_label="low",
            ambiguity_detected=True,
            ambiguity_reason=f"invalid_query:{query_validity.status}",
            no_resolution=True,
            no_resolution_reason=f"invalid_query:{query_validity.status}",
        )

    clusters = _cluster_identity_evidence(ranked, query_name, context)
    winning_cluster = clusters[0]
    top = winning_cluster.representative
    second_cluster = clusters[1] if len(clusters) > 1 else None
    second = second_cluster.representative if second_cluster else None
    content, fetch_status = extract_top_candidate_content(top.result.url, fetcher=fetcher)
    normalized_name, role, organization, inferred_location = infer_profile_signals(content)
    canonical_name, canonical_name_quality, canonical_name_source = _derive_candidate_name(content)
    if not canonical_name:
        fallback_title = _clean_title_fragment(top.result.title)
        if _looks_person_like_text(fallback_title):
            canonical_name = normalize_name(fallback_title)
            canonical_name_quality = 0.62
            canonical_name_source = "search_title_fallback"
    if canonical_name:
        normalized_name = canonical_name

    same_person_probability = min(1.0, top.score * {"full_match": 1.0, "reordered_match": 0.9, "partial_match": 0.7, "weak_match": 0.4}[top.name_match])
    context_probability = min(1.0, top.context_strength + (0.15 if role or organization or inferred_location else 0.0))

    if fetch_status == "ok":
        if content.main_text or content.meta_description:
            resolution_path: Literal["full_content", "partial_content", "search_only", "fetch_blocked"] = "full_content"
        else:
            resolution_path = "partial_content"
    elif fetch_status.startswith("http_error:"):
        resolution_path = "fetch_blocked"
    else:
        resolution_path = "search_only"

    score_gap = winning_cluster.aggregate_score - second_cluster.aggregate_score if second_cluster else 1.0
    source_diversity = len({item.result.domain for item in ranked[:5]})
    has_context_constraints = any(
        [
            context.role,
            context.organization,
            context.location,
            context.domain_activity,
            context.media_platform,
            context.institutional_hint,
            context.generic_terms,
            context.expected_domains,
        ]
    )
    weak_top_evidence = (
        winning_cluster.aggregate_score < 0.52
        or top.name_match == "weak_match"
        or top.entity_type in {"unknown", "generic_article", "aggregator_profile", "directory_listing"}
        or top.authority_tier in {"directory_aggregator", "low_authority_bio_seo", "junk_or_malformed"}
    )
    close_competition = second_cluster is not None and score_gap < 0.1 and second_cluster.aggregate_score > 0.44
    common_name_competition = (
        not has_context_constraints
        and second is not None
        and top.name_match in {"full_match", "reordered_match"}
        and second.name_match in {"full_match", "reordered_match", "partial_match"}
        and score_gap < 0.22
    )
    official_superiority = (
        top.authority_tier == "official_institutional"
        and top.name_match in {"full_match", "reordered_match"}
        and top.context_strength >= 0.18
        and score_gap >= 0.1
    )
    evidence_strength = winning_cluster.aggregate_score - query_validity.penalty
    if winning_cluster.independent_domains >= 2:
        evidence_strength += 0.08
    elif source_diversity >= 3:
        evidence_strength += 0.05
    if second is not None and score_gap >= 0.16:
        evidence_strength += 0.04
    evidence_strength += min(0.08, top.context_strength * 0.15)
    evidence_strength += 0.08 * top.typing_confidence
    evidence_strength += 0.12 * canonical_name_quality
    if official_superiority:
        evidence_strength += 0.08
    distinctiveness = _query_distinctiveness(query_name)
    if top.authority_tier in {"official_institutional", "strong_encyclopedic"} and top.name_match in {"full_match", "reordered_match"}:
        distinctiveness += 0.14
    if second_cluster is None or score_gap >= 0.16:
        distinctiveness += 0.08
    distinctiveness = min(1.0, distinctiveness)
    evidence_strength += min(0.16, 0.18 * distinctiveness)
    ambiguity_detected = close_competition or common_name_competition
    ambiguity_reason = "multiple_plausible_candidates" if ambiguity_detected else None
    if second is not None and top.authority_tier == "official_institutional":
        authority_order = {
            "official_institutional": 6,
            "public_structured_profile": 5,
            "reputable_media": 4,
            "strong_encyclopedic": 3,
            "low_authority_bio_seo": 2,
            "directory_aggregator": 1,
            "junk_or_malformed": 0,
        }
        if (
            authority_order[top.authority_tier] - authority_order[second.authority_tier] >= 1
            and top.context_strength >= second.context_strength
            and top.name_match in {"full_match", "reordered_match"}
        ):
            ambiguity_detected = False
            ambiguity_reason = None

    strong_top_profile = (
        winning_cluster.aggregate_score >= 0.62
        and top.name_match in {"full_match", "reordered_match"}
        and top.entity_type in {"official_profile", "official_bio", "institutional_profile", "person_profile", "reference_entry"}
    )
    strong_single_source_resolution = (
        winning_cluster.independent_domains == 1
        and top.name_match in {"full_match", "reordered_match"}
        and canonical_name_quality >= 0.55
        and top.authority_tier in {"official_institutional", "strong_encyclopedic", "public_structured_profile"}
        and top.entity_type not in {"directory_listing", "aggregator_profile", "generic_article", "unknown"}
        and top.context_strength >= 0.15
        and not (second_cluster is not None and score_gap < 0.08)
        and distinctiveness >= 0.45
    )
    insufficient_evidence = (evidence_strength < 0.56 and not strong_top_profile) or weak_top_evidence
    if strong_single_source_resolution and weak_top_evidence is False:
        insufficient_evidence = False
    confidence_label: Literal["high", "medium", "low"] = "high"
    if insufficient_evidence:
        confidence_label = "low"
    elif evidence_strength < 0.75 or (second is not None and score_gap < 0.16):
        confidence_label = "medium"
    query_token_count = len(_normalized_tokens(query_name))
    if (
        not has_context_constraints
        and query_validity.status == "valid"
        and query_token_count <= 2
        and second_cluster is not None
        and top.authority_tier not in {"official_institutional"}
    ):
        confidence_label = "low"
        insufficient_evidence = True

    explicit_org_alignment = bool(
        context.organization
        and normalize_name(context.organization) in _joined_text(top.result)
        and top.authority_tier == "official_institutional"
    )
    if official_superiority and not ambiguity_detected and confidence_label == "medium" and evidence_strength >= 0.62:
        confidence_label = "high"
    if explicit_org_alignment and top.name_match in {"full_match", "reordered_match"} and score_gap >= 0.1:
        confidence_label = "high"
    if (
        top.authority_tier == "official_institutional"
        and top.score >= 0.9
        and top.name_match in {"full_match", "reordered_match"}
    ):
        confidence_label = "high"

    only_single_cluster_source = (
        len(winning_cluster.candidates) == 1
        or winning_cluster.independent_domains <= 1
    )
    single_encyclopedic_fallback = top.authority_tier == "strong_encyclopedic" and only_single_cluster_source
    if only_single_cluster_source and not (
        top.authority_tier == "official_institutional"
        and top.entity_type in _PRIMARY_OFFICIAL_ENTITY_TYPES
        and top.name_match in {"full_match", "reordered_match"}
    ):
        if confidence_label == "high":
            confidence_label = "medium"
        evidence_strength = min(evidence_strength, 0.72)
    if single_encyclopedic_fallback:
        if distinctiveness >= 0.62 and top.name_match in {"full_match", "reordered_match"} and top.context_strength >= 0.12:
            confidence_label = "medium"
        else:
            confidence_label = "low"

    wikipedia_fallback_without_context = (
        top.result.domain.endswith("wikipedia.org")
        and top.context_strength < 0.2
        and (second is None or score_gap < 0.2)
        and query_validity.status == "valid"
        and distinctiveness < 0.48
    )
    if wikipedia_fallback_without_context:
        return ResolutionOutput(
            normalized_candidate_name="",
            source_url="",
            final_score=winning_cluster.aggregate_score,
            entity_type=top.entity_type,
            same_person_probability=0.0,
            context_match_probability=top.context_strength,
            possible_role=None,
            possible_organization=None,
            possible_location=None,
            explanation=(
                "NO_RESOLUTION: generic encyclopedic fallback without context support; "
                f"score={top.score:.3f}; score_gap={score_gap:.3f}; source_diversity={source_diversity}"
            ),
            resolution_path=resolution_path,
            fetch_status=fetch_status,
            confidence_label="low",
            ambiguity_detected=True,
            ambiguity_reason="generic_fallback_without_context_support",
            no_resolution=True,
            no_resolution_reason="generic_encyclopedic_fallback_without_support",
        )

    common_name_structured_profile_only = (
        top.result.domain.endswith(tuple(_STRUCTURED_PROFILE_DOMAINS))
        and winning_cluster.independent_domains == 1
        and winning_cluster.official_support_count == 0
        and winning_cluster.strong_source_count <= 1
    )
    if common_name_structured_profile_only and top.context_strength < 0.42:
        return ResolutionOutput(
            normalized_candidate_name="",
            source_url="",
            final_score=winning_cluster.aggregate_score,
            entity_type=top.entity_type,
            same_person_probability=same_person_probability,
            context_match_probability=context_probability,
            possible_role=role,
            possible_organization=organization,
            possible_location=inferred_location,
            explanation="NO_RESOLUTION: common-name structured profile without independent corroboration.",
            resolution_path=resolution_path,
            fetch_status=fetch_status,
            confidence_label="low",
            ambiguity_detected=False,
            ambiguity_reason=None,
            no_resolution=True,
            no_resolution_reason="common_name_structured_profile_without_corroboration",
        )

    if insufficient_evidence and not ambiguity_detected:
        return ResolutionOutput(
            normalized_candidate_name="",
            source_url="",
            final_score=winning_cluster.aggregate_score,
            entity_type=top.entity_type,
            same_person_probability=same_person_probability,
            context_match_probability=context_probability,
            possible_role=role,
            possible_organization=organization,
            possible_location=inferred_location,
            explanation=(
                "NO_RESOLUTION: insufficient absolute evidence for unique identity; "
                f"score={top.score:.3f}; evidence={evidence_strength:.3f}; canonical_name_quality={canonical_name_quality:.2f}"
            ),
            resolution_path=resolution_path,
            fetch_status=fetch_status,
            confidence_label="low",
            ambiguity_detected=False,
            ambiguity_reason=None,
            no_resolution=True,
            no_resolution_reason="insufficient_evidence",
        )

    prefix = "AMBIGUOUS: " if ambiguity_detected else ""
    explanation = (
        f"{prefix}Selected {top.result.domain} with {top.name_match} and {top.entity_type}; "
        f"authority={top.authority_tier}; seo_penalty={top.seo_penalty:.2f}; "
        f"query_validity={query_validity.status}; score_cap={top.score_cap_applied if top.score_cap_applied is not None else 'none'}; "
        f"official_superiority_bonus={'yes' if official_superiority else 'no'}; "
        f"typing_confidence={top.typing_confidence:.2f}; "
        f"canonical_name_quality={canonical_name_quality:.2f}; canonical_name_source={canonical_name_source}; "
        f"name_source={'derived' if normalized_name else 'query_fallback'}; "
        f"signals: {', '.join(top.reasons[:8])}; "
        f"cluster_key={winning_cluster.key}; cluster_sources={len(winning_cluster.candidates)}; "
        f"cluster_domains={winning_cluster.independent_domains}; "
        f"path={resolution_path}; fetch={fetch_status}; score_gap={score_gap:.3f}; "
        f"evidence={evidence_strength:.3f}; confidence={confidence_label}"
    )

    return ResolutionOutput(
        normalized_candidate_name=normalized_name or normalize_name(query_name),
        source_url=top.result.url,
        final_score=top.score,
        entity_type=top.entity_type,
        same_person_probability=same_person_probability,
        context_match_probability=context_probability,
        possible_role=role,
        possible_organization=organization,
        possible_location=inferred_location,
        resolution_path=resolution_path,
        fetch_status=fetch_status,
        explanation=explanation,
        confidence_label=confidence_label,
        ambiguity_detected=ambiguity_detected,
        ambiguity_reason=ambiguity_reason,
        no_resolution=ambiguity_detected,
        no_resolution_reason=ambiguity_reason if ambiguity_detected else None,
    )
