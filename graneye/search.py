from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from typing import Any, Mapping
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

_DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
_DDG_INSTANT_ENDPOINT = "https://api.duckduckgo.com/"
_DDG_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
_RESULT_BLOCK_PATTERN = re.compile(
    r"<(article|div)\b[^>]*\bclass=(['\"])[^'\"]*\bresult\b[^'\"]*\2[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_RESULT_LINK_PATTERN = re.compile(
    r"<a\b[^>]*\bclass=(['\"])[^'\"]*\bresult__a\b[^'\"]*\1[^>]*\bhref=(['\"])(.*?)\2[^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_RESULT_SNIPPET_PATTERN = re.compile(
    r"<(a|div|span)\b[^>]*\bclass=(['\"])[^'\"]*\b(result__snippet|snippet)\b[^'\"]*\2[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_FALLBACK_RESULT_LINK_PATTERN = re.compile(
    r"<a\b[^>]*\bhref=(['\"])(.*?)\1[^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WS_PATTERN = re.compile(r"\s+")
_SEARCH_ENGINE_HOSTS = {
    "duckduckgo.com",
    "links.duckduckgo.com",
    "google.com",
    "bing.com",
    "search.yahoo.com",
}
_DDG_INTERNAL_PATH_HINTS = (
    "/privacy",
    "/help",
    "/settings",
    "/about",
    "/app",
    "/bang",
    "/traffic",
)
_PLACEHOLDER_TITLES = {"here", "cached", "feedback", "more results"}


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Normalized search result used by downstream ranking logic."""

    title: str
    url: str
    domain: str
    snippet: str | None = None


@dataclass(slots=True, frozen=True)
class FilterDecision:
    """Decision trace for filtering a normalized search result."""

    result: SearchResult
    accepted: bool
    reason: str


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_html_text(value: str) -> str:
    without_tags = _TAG_PATTERN.sub(" ", value)
    normalized = _WS_PATTERN.sub(" ", unescape(without_tags))
    return normalized.strip()


def _resolve_ddg_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc and "duckduckgo.com" not in parsed.netloc:
        return url

    if parsed.path != "/l/":
        return url

    query = parse_qs(parsed.query)
    uddg_values = query.get("uddg")
    if not uddg_values:
        return url
    return unescape(unquote(uddg_values[0])).strip()


def _is_likely_web_result(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.casefold().removeprefix("www.")
    if not host:
        return False
    if host.endswith("duckduckgo.com") and parsed.path.startswith("/y.js"):
        return False
    return True


def _is_non_trivial_title(title: str) -> bool:
    normalized = _WS_PATTERN.sub(" ", title).strip()
    if len(normalized) < 4:
        return False
    lowered = normalized.casefold()
    if lowered in _PLACEHOLDER_TITLES:
        return False
    return len(re.findall(r"[a-zA-ZÀ-ÿ]", normalized)) >= 3


def _is_internal_or_search_engine_page(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().removeprefix("www.")
    if not host:
        return True

    if host in _SEARCH_ENGINE_HOSTS:
        if host.endswith("duckduckgo.com"):
            if parsed.path == "/l/":
                return False
            path = parsed.path.casefold() or "/"
            if path == "/" or any(path.startswith(prefix) for prefix in _DDG_INTERNAL_PATH_HINTS):
                return True
            if parsed.query:
                return True
        return True

    return False


def _is_candidate_worthy_result(title: str, url: str, snippet: str = "") -> bool:
    return evaluate_candidate_result(title, url, snippet).accepted


def evaluate_candidate_result(title: str, url: str, snippet: str = "") -> FilterDecision:
    result = SearchResult(
        title=title,
        url=url,
        domain=urlparse(url).netloc.casefold().removeprefix("www.") if url else "",
        snippet=snippet or None,
    )
    if not url:
        return FilterDecision(result=result, accepted=False, reason="missing_url")
    if not _is_likely_web_result(url):
        return FilterDecision(result=result, accepted=False, reason="non_http_url")
    if _is_internal_or_search_engine_page(url):
        return FilterDecision(result=result, accepted=False, reason="search_engine_or_internal")
    if not _is_non_trivial_title(title):
        return FilterDecision(result=result, accepted=False, reason="trivial_title")
    if not urlparse(url).netloc:
        return FilterDecision(result=result, accepted=False, reason="missing_domain")
    text = " ".join(part for part in [title, snippet] if part).casefold()
    if any(token in text for token in ("duckduckgo", "privacy", "protection", "safe search")):
        return FilterDecision(result=result, accepted=False, reason="search_engine_noise_text")
    return FilterDecision(result=result, accepted=True, reason="accepted")


def parse_duckduckgo_html_results(html: str, *, max_results: int = 10) -> list[dict[str, str]]:
    """Extract DuckDuckGo HTML results in a resilient way without relying on parser state."""

    results: list[dict[str, str]] = []

    seen_urls: set[str] = set()

    for block_match in _RESULT_BLOCK_PATTERN.finditer(html):
        block = block_match.group(3)
        link_match = _RESULT_LINK_PATTERN.search(block)
        if not link_match:
            continue

        raw_url = unescape(link_match.group(3).strip())
        url = _resolve_ddg_redirect(raw_url)
        title = _clean_html_text(link_match.group(4))

        snippet_match = _RESULT_SNIPPET_PATTERN.search(block)
        snippet = _clean_html_text(snippet_match.group(4)) if snippet_match else ""

        if not url or url in seen_urls:
            continue
        if not _is_likely_web_result(url):
            continue
        if not _is_non_trivial_title(title):
            continue

        seen_urls.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break

    if len(results) >= max_results:
        return results

    # Fallback extraction for HTML layout variants where result wrappers/classes differ.
    for link_match in _FALLBACK_RESULT_LINK_PATTERN.finditer(html):
        raw_url = unescape(link_match.group(2).strip())
        url = _resolve_ddg_redirect(raw_url)
        title = _clean_html_text(link_match.group(3))
        if not title or not url or url in seen_urls:
            continue

        if not _is_likely_web_result(url):
            continue
        if not _is_non_trivial_title(title):
            continue

        seen_urls.add(url)
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= max_results:
            break

    return results


def search_duckduckgo_html(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    """Fetch and parse DuckDuckGo HTML results for a query."""

    encoded = quote_plus(query)
    url = f"{_DDG_HTML_ENDPOINT}?q={encoded}"
    request = Request(
        url,
        headers={
            "User-Agent": _DDG_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://duckduckgo.com/",
        },
    )

    with urlopen(request, timeout=10) as response:
        html = response.read(600_000).decode("utf-8", errors="ignore")

    return parse_duckduckgo_html_results(html, max_results=max_results)


def search_duckduckgo_instant_answer(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    encoded = quote_plus(query)
    url = f"{_DDG_INSTANT_ENDPOINT}?q={encoded}&format=json&no_html=1&skip_disambig=1"
    request = Request(url, headers={"User-Agent": _DDG_USER_AGENT, "Accept": "application/json"})

    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    normalized: list[dict[str, str]] = []

    abstract_url = str(payload.get("AbstractURL", "")).strip()
    if abstract_url:
        abstract_title = str(payload.get("Heading", "")).strip() or query
        abstract_snippet = str(payload.get("AbstractText", "")).strip()
        if _is_candidate_worthy_result(abstract_title, abstract_url, abstract_snippet):
            normalized.append(
                {
                    "title": abstract_title,
                    "url": abstract_url,
                    "snippet": abstract_snippet,
                }
            )

    def append_topic(item: dict[str, object]) -> None:
        topic_url = str(item.get("FirstURL", "")).strip()
        text = str(item.get("Text", "")).strip()
        if not topic_url:
            return
        title = text.split(" - ")[0].strip() if text else ""
        if _is_candidate_worthy_result(title, topic_url, text):
            normalized.append({"title": title, "url": topic_url, "snippet": text})

    for topic in payload.get("RelatedTopics", []):
        if isinstance(topic, dict) and "FirstURL" in topic:
            append_topic(topic)
        elif isinstance(topic, dict):
            for child in topic.get("Topics", []):
                if isinstance(child, dict):
                    append_topic(child)

    return normalized[:max_results]


def normalize_search_result(payload: Mapping[str, Any]) -> SearchResult:
    """Normalize a raw search payload into a stable SearchResult object."""

    title = _to_text(payload.get("title"))
    snippet_raw = _to_text(payload.get("snippet") or payload.get("description"))
    url = _to_text(payload.get("url") or payload.get("link"))

    parsed = urlparse(url)
    domain = parsed.netloc.casefold().removeprefix("www.") if parsed.netloc else ""

    return SearchResult(
        title=title,
        url=url,
        domain=domain,
        snippet=snippet_raw or None,
    )


def normalize_search_results(raw_results: list[Mapping[str, Any]]) -> list[SearchResult]:
    return [normalize_search_result(item) for item in raw_results]


def filter_search_results(results: list[SearchResult]) -> tuple[list[SearchResult], list[FilterDecision]]:
    filtered: list[SearchResult] = []
    decisions: list[FilterDecision] = []
    seen_urls: set[str] = set()
    for result in results:
        decision = evaluate_candidate_result(result.title, result.url, result.snippet or "")
        if decision.accepted and result.url in seen_urls:
            decision = FilterDecision(result=result, accepted=False, reason="duplicate_url")
        if decision.accepted:
            filtered.append(result)
            seen_urls.add(result.url)
        decisions.append(FilterDecision(result=result, accepted=decision.accepted, reason=decision.reason))
    return filtered, decisions


def enrich_search_results(raw_results: list[Mapping[str, Any]]) -> list[SearchResult]:
    """Normalize search results while preserving ordering and missing snippets."""

    normalized = normalize_search_results(raw_results)
    filtered, _ = filter_search_results(normalized)
    return filtered
