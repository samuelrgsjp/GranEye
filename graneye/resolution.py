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

EntityType = Literal["person_profile", "directory", "company_page", "article", "unknown"]
NameMatchQuality = Literal["full_match", "reordered_match", "partial_match", "weak_match"]

_DIRECTORY_HOST_HINTS = {"zoominfo", "rocketreach", "spokeo", "beenverified", "whitepages"}
_COMPANY_HINTS = {"about", "company", "team", "leadership", "careers"}
_ARTICLE_HINTS = {"news", "blog", "article", "press"}
_PROFILE_SEGMENTS = {"in", "u", "user", "profile", "people"}


@dataclass(slots=True, frozen=True)
class ContextQuery:
    """Optional context to disambiguate person identity resolution."""

    profession: str | None = None
    location: str | None = None
    expected_domains: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ScoredCandidate:
    """Scored candidate with deterministic evidence and debugging details."""

    result: SearchResult
    score: float
    entity_type: EntityType
    name_match: NameMatchQuality
    context_strength: float
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


def _normalized_tokens(value: str) -> list[str]:
    normalized = normalize_name(value)
    return [token for token in normalized.split(" ") if token]


def _joined_text(result: SearchResult) -> str:
    return " ".join(part for part in [result.title, result.snippet or "", result.url] if part).casefold()


def detect_entity_type(result: SearchResult) -> EntityType:
    """Classify result as profile/directory/company/article/unknown."""

    parsed = urlparse(result.url)
    path_segments = [segment.casefold() for segment in parsed.path.split("/") if segment]

    if is_directory_url(result.url) or any(token in result.domain for token in _DIRECTORY_HOST_HINTS):
        return "directory"

    if any(segment in _ARTICLE_HINTS for segment in path_segments):
        return "article"

    if any(segment in _COMPANY_HINTS for segment in path_segments):
        return "company_page"

    if len(path_segments) >= 2 and path_segments[-2] in _PROFILE_SEGMENTS:
        return "person_profile"

    if re.search(r"\b(profile|bio|about\s+me)\b", _joined_text(result)):
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
    """Compute context match score using profession, location, and domain preferences."""

    reasons: list[str] = []
    score = 0.0
    haystack = _joined_text(result)

    if context.profession:
        profession = normalize_name(context.profession)
        if profession and profession in haystack:
            score += 0.45
            reasons.append("profession_match")

    if context.location:
        location = normalize_name(context.location)
        if location and location in haystack:
            score += 0.35
            reasons.append("location_match")

    if context.expected_domains and any(result.domain.endswith(domain) for domain in context.expected_domains):
        score += 0.2
        reasons.append("domain_relevance")

    return min(score, 1.0), tuple(reasons)


def is_noise_result(result: SearchResult) -> bool:
    """Detect low-signal pages such as directories, list pages, and SEO aggregators."""

    text = _joined_text(result)
    noisy_phrases = ("top ", "best ", "list of", "find people", "people search")

    return (
        detect_entity_type(result) == "directory"
        or any(phrase in text for phrase in noisy_phrases)
        or re.search(r"\b(aggregator|directory|listing)\b", text) is not None
    )


def score_candidate(result: SearchResult, query_name: str, context: ContextQuery) -> ScoredCandidate:
    """Deterministic scoring with explicit bonuses/penalties for explainability."""

    reasons: list[str] = []
    entity_type = detect_entity_type(result)
    name_match = detect_name_match_quality(query_name, result)
    context_strength, context_reasons = context_match_strength(result, context)
    noisy = is_noise_result(result)

    score = 0.0

    entity_weights: dict[EntityType, float] = {
        "person_profile": 0.35,
        "company_page": 0.1,
        "article": 0.05,
        "directory": -0.25,
        "unknown": 0.0,
    }
    score += entity_weights[entity_type]
    reasons.append(f"entity:{entity_type}")

    name_weights: dict[NameMatchQuality, float] = {
        "full_match": 0.4,
        "reordered_match": 0.3,
        "partial_match": 0.15,
        "weak_match": 0.02,
    }
    score += name_weights[name_match]
    reasons.append(f"name:{name_match}")

    snippet_bonus = 0.0
    if result.snippet:
        snippet_terms = set(_normalized_tokens(result.snippet))
        query_terms = set(_normalized_tokens(query_name))
        if query_terms:
            snippet_bonus = 0.15 * (len(snippet_terms & query_terms) / len(query_terms))
            score += snippet_bonus
    reasons.append(f"snippet_bonus:{snippet_bonus:.2f}")

    score += 0.3 * context_strength
    reasons.extend(context_reasons)

    has_context_constraints = any([context.profession, context.location, context.expected_domains])
    if has_context_constraints and name_match in {"full_match", "reordered_match"} and context_strength < 0.12:
        score -= 0.22
        reasons.append("weak_context_exact_name_penalty")

    if entity_type == "person_profile":
        score += 0.08
        reasons.append("profile_structure_bonus")

    if noisy:
        score -= 0.35
        reasons.append("noise_penalty")

    return ScoredCandidate(
        result=result,
        score=max(0.0, min(1.0, score)),
        entity_type=entity_type,
        name_match=name_match,
        context_strength=context_strength,
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

    role_match = re.search(r"\b(engineer|developer|manager|director|analyst|researcher|journalist|designer)\b", merged, re.I)
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
    profession: str | None = None,
    location: str | None = None,
    expected_domains: tuple[str, ...] = (),
    fetcher: Callable[[str], str] | None = None,
) -> ResolutionOutput | None:
    """Resolve the highest-confidence identity candidate and extract top-page signals."""

    context = ContextQuery(profession=profession, location=location, expected_domains=expected_domains)
    ranked = rank_candidates(results, query_name, context)
    if not ranked:
        return None

    top = ranked[0]
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

    explanation = (
        f"Selected {top.result.domain} with {top.name_match} and {top.entity_type}; "
        f"signals: {', '.join(top.reasons[:5])}; path={resolution_path}; fetch={fetch_status}"
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
    )
