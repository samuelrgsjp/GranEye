from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Iterable, Literal
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .detection import is_directory_url
from .normalization import normalize_name
from .search import SearchResult

EntityType = Literal[
    "person_profile",
    "official_profile",
    "official_bio",
    "academic_profile",
    "institutional_profile",
    "media_profile",
    "creator_profile",
    "article",
    "directory",
    "aggregator",
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

_DIRECTORY_HOST_HINTS = {"zoominfo", "rocketreach", "spokeo", "beenverified", "whitepages"}
_COMPANY_HINTS = {"about", "company", "team", "leadership", "careers", "executive", "management"}
_ARTICLE_HINTS = {"news", "blog", "article", "press"}
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


@dataclass(slots=True, frozen=True)
class TopCandidateContent:
    """Lightweight extraction from the top-ranked candidate page."""

    page_title: str
    meta_description: str
    headings: tuple[str, ...]
    main_text: str


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


@dataclass(slots=True, frozen=True)
class QueryValidityAssessment:
    status: Literal["valid", "too_short", "too_generic", "non_person_like", "numeric_or_garbage"]
    penalty: float
    reasons: tuple[str, ...]


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

    parsed = urlparse(result.url)
    domain = parsed.netloc.casefold().removeprefix("www.") or result.domain.casefold()
    path_segments = [segment.casefold() for segment in parsed.path.split("/") if segment]

    joined_text = _joined_text(result)
    tail_segment = path_segments[-1] if path_segments else ""

    if is_directory_url(result.url) or any(token in domain for token in _DIRECTORY_HOST_HINTS):
        return "directory"
    if any(token in domain for token in _AGGREGATOR_HINTS):
        return "aggregator"

    if any(segment in _ARTICLE_HINTS for segment in path_segments):
        return "article"

    if any(domain.endswith(platform_domain) for platform_domain in _CREATOR_PLATFORM_DOMAINS):
        if any(segment in path_segments for segment in {"channel", "c", "user", "creator"}) or any(
            segment.startswith("@") for segment in path_segments
        ):
            return "creator_profile"
        return "media_profile"

    if any(domain.endswith(media_domain) for media_domain in _MEDIA_DOMAIN_HINTS):
        return "media_profile"

    if len(path_segments) >= 2 and path_segments[-2] in _PROFILE_SEGMENTS:
        return "person_profile"

    if any(hint in domain for hint in _ACADEMIC_DOMAIN_HINTS) or any(
        segment in path_segments for segment in {"faculty", "research", "department", "academics", "professor"}
    ):
        if any(segment in path_segments for segment in {"faculty", "professor", "staff"}):
            return "academic_profile"
        return "institutional_profile"

    if len(path_segments) >= 2 and path_segments[-2] in _OFFICIAL_PROFILE_SEGMENTS:
        if re.search(r"[-_]", tail_segment):
            return "official_profile"
        if any(token in joined_text for token in ("professor", "attorney", "psychologist", "md", "phd", "chief", "ceo")):
            return "official_profile"
        return "institutional_profile"

    if any(segment in _COMPANY_HINTS for segment in path_segments):
        return "official_bio"

    if re.search(r"\b(profile|bio|about\s+me|executive\s+profile|faculty|attorney|psychologist)\b", joined_text):
        return "person_profile"

    return "unknown"


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
        detect_entity_type(result) in {"directory", "aggregator"}
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
        return "official_institutional", 0.24, ["official_or_institutional_domain"]
    if any(domain.endswith(media) for media in _REPUTABLE_MEDIA_HINTS):
        return "reputable_media", 0.16, ["reputable_media_domain"]
    if any(domain.endswith(network) for network in _NETWORK_PROFILE_DOMAINS):
        return "public_structured_profile", 0.08, ["structured_public_profile"]
    if detect_entity_type(result) in {"directory", "aggregator"}:
        return "directory_aggregator", -0.2, ["directory_or_aggregator_source"]
    if any(hint in text for hint in _LOW_AUTHORITY_BIO_HINTS):
        return "low_authority_bio_seo", -0.24, ["low_authority_biography_patterns"]
    return "public_structured_profile", 0.02, ["default_public_profile_assumption"]


def _seo_bio_penalty(result: SearchResult) -> tuple[float, list[str]]:
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
    if detect_entity_type(result) in {"article", "unknown"} and "biography" in text and result.domain.count(".") >= 1:
        penalty += 0.06
        reasons.append("generic_biography_page_penalty")
    return min(0.3, penalty), reasons


def score_candidate(result: SearchResult, query_name: str, context: ContextQuery) -> ScoredCandidate:
    """Deterministic scoring with explicit bonuses/penalties for explainability."""

    reasons: list[str] = []
    entity_type = detect_entity_type(result)
    name_match = detect_name_match_quality(query_name, result)
    context_strength, context_reasons = context_match_strength(result, context)
    authority_tier, authority_weight, authority_reasons = detect_source_authority(result)
    seo_penalty, seo_reasons = _seo_bio_penalty(result)
    noisy = is_noise_result(result)

    score = 0.0

    entity_weights: dict[EntityType, float] = {
        "person_profile": 0.35,
        "official_profile": 0.33,
        "official_bio": 0.24,
        "academic_profile": 0.3,
        "institutional_profile": 0.2,
        "media_profile": 0.22,
        "creator_profile": 0.24,
        "article": 0.05,
        "directory": -0.25,
        "aggregator": -0.2,
        "unknown": 0.0,
    }
    score += entity_weights[entity_type]
    reasons.append(f"entity:{entity_type}")

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

    if entity_type in {"person_profile", "official_profile", "academic_profile", "creator_profile"}:
        score += 0.08
        reasons.append("profile_structure_bonus")

    if context.role and entity_type in {"official_profile", "official_bio"}:
        score += 0.06
        reasons.append("role_official_alignment_bonus")
    if context.media_platform and entity_type in {"creator_profile", "media_profile"}:
        score += 0.08
        reasons.append("creator_media_alignment_bonus")
    if context.institutional_hint and entity_type in {"academic_profile", "institutional_profile"}:
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

    if entity_type == "article" and has_context_constraints and context_strength < 0.22:
        score -= 0.12
        reasons.append("article_low_context_penalty")

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

    return ScoredCandidate(
        result=result,
        score=max(0.0, min(1.0, score)),
        entity_type=entity_type,
        name_match=name_match,
        context_strength=context_strength,
        authority_tier=authority_tier,
        seo_penalty=seo_penalty,
        is_noise=noisy,
        reasons=tuple(reasons),
    )


class _HTMLSignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._in_heading: str | None = None
        self._in_ignored = False
        self.title = ""
        self.meta_description = ""
        self.headings: list[str] = []
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        attrs_dict = {key.casefold(): (value or "") for key, value in attrs}
        if lowered in {"script", "style", "noscript"}:
            self._in_ignored = True
        if lowered == "title":
            self._in_title = True
        if lowered in {"h1", "h2"}:
            self._in_heading = lowered
        if lowered == "meta" and attrs_dict.get("name", "").casefold() == "description":
            self.meta_description = attrs_dict.get("content", "").strip()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript"}:
            self._in_ignored = False
        if lowered == "title":
            self._in_title = False
        if lowered in {"h1", "h2"}:
            self._in_heading = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._in_ignored:
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
        return TopCandidateContent(page_title="", meta_description="", headings=(), main_text=""), status
    except URLError:
        return TopCandidateContent(page_title="", meta_description="", headings=(), main_text=""), "network_error"
    except TimeoutError:
        return TopCandidateContent(page_title="", meta_description="", headings=(), main_text=""), "timeout"
    except Exception:
        return TopCandidateContent(page_title="", meta_description="", headings=(), main_text=""), "fetch_error"

    if not html:
        return TopCandidateContent(page_title="", meta_description="", headings=(), main_text=""), "empty_response"

    parser = _HTMLSignalParser()
    parser.feed(html)

    return TopCandidateContent(
        page_title=parser.title.strip(),
        meta_description=parser.meta_description.strip(),
        headings=tuple(parser.headings[:8]),
        main_text=parser.main_text[:2500].strip(),
    ), "ok"


def infer_profile_signals(content: TopCandidateContent) -> tuple[str, str | None, str | None, str | None]:
    """Infer normalized name, profession, organization, and location from page content."""

    merged = " | ".join(
        part for part in [content.page_title, content.meta_description, " ".join(content.headings), content.main_text] if part
    )
    normalized_name = normalize_name(content.page_title.split("|")[0]) if content.page_title else ""

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


def rank_candidates(results: Iterable[SearchResult], query_name: str, context: ContextQuery) -> list[ScoredCandidate]:
    """Rank all candidates deterministically using evidence-driven heuristics."""

    scored = [score_candidate(result, query_name, context) for result in results]
    return sorted(scored, key=lambda item: (-item.score, item.result.domain, item.result.url))


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

    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    content, fetch_status = extract_top_candidate_content(top.result.url, fetcher=fetcher)
    normalized_name, role, organization, inferred_location = infer_profile_signals(content)

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

    score_gap = top.score - second.score if second else 1.0
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
    low_evidence = (
        top.score < 0.45
        or top.name_match == "weak_match"
        or top.entity_type in {"unknown", "article", "aggregator"}
        or top.authority_tier in {"directory_aggregator", "low_authority_bio_seo", "junk_or_malformed"}
    )
    close_competition = second is not None and score_gap < 0.08 and second.score > 0.33
    common_name_competition = (
        not has_context_constraints
        and second is not None
        and top.name_match in {"full_match", "reordered_match"}
        and second.name_match in {"full_match", "reordered_match", "partial_match"}
        and score_gap < 0.22
    )
    ambiguity_detected = low_evidence or close_competition or common_name_competition or query_validity.status != "valid"
    ambiguity_reason = None
    evidence_strength = top.score - query_validity.penalty
    if source_diversity >= 3:
        evidence_strength += 0.05
    if second is not None and score_gap >= 0.18:
        evidence_strength += 0.04
    confidence_label: Literal["high", "medium", "low"] = "high"
    if ambiguity_detected or evidence_strength < 0.48:
        confidence_label = "low"
        if query_validity.status != "valid":
            ambiguity_reason = f"invalid_query:{query_validity.status}"
        elif common_name_competition:
            ambiguity_reason = "multiple_plausible_candidates"
        else:
            ambiguity_reason = "weak_evidence" if low_evidence else "multiple_plausible_candidates"
    elif evidence_strength < 0.68 or (second is not None and score_gap < 0.16):
        confidence_label = "medium"
    query_token_count = len(_normalized_tokens(query_name))
    if (
        not has_context_constraints
        and query_validity.status == "valid"
        and query_token_count <= 2
        and top.authority_tier not in {"official_institutional", "strong_encyclopedic"}
    ):
        confidence_label = "low"
        ambiguity_detected = True
        ambiguity_reason = ambiguity_reason or "multiple_plausible_candidates"

    prefix = "AMBIGUOUS: " if ambiguity_detected else ""
    explanation = (
        f"{prefix}Selected {top.result.domain} with {top.name_match} and {top.entity_type}; "
        f"authority={top.authority_tier}; seo_penalty={top.seo_penalty:.2f}; "
        f"query_validity={query_validity.status}; signals: {', '.join(top.reasons[:6])}; "
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
    )
