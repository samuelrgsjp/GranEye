from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from typing import Any, Mapping
from urllib.parse import parse_qs, quote_plus, urlparse
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
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WS_PATTERN = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Normalized search result used by downstream ranking logic."""

    title: str
    url: str
    domain: str
    snippet: str | None = None


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
    if parsed.path != "/l/":
        return url

    query = parse_qs(parsed.query)
    uddg_values = query.get("uddg")
    if not uddg_values:
        return url
    return unescape(uddg_values[0])


def parse_duckduckgo_html_results(html: str, *, max_results: int = 10) -> list[dict[str, str]]:
    """Extract DuckDuckGo HTML results in a resilient way without relying on parser state."""

    results: list[dict[str, str]] = []

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

        if not url:
            continue

        results.append({"title": title, "url": url, "snippet": snippet})
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
        normalized.append(
            {
                "title": str(payload.get("Heading", "")).strip() or query,
                "url": abstract_url,
                "snippet": str(payload.get("AbstractText", "")).strip(),
            }
        )

    def append_topic(item: dict[str, object]) -> None:
        topic_url = str(item.get("FirstURL", "")).strip()
        text = str(item.get("Text", "")).strip()
        if not topic_url:
            return
        title = text.split(" - ")[0].strip() if text else ""
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


def enrich_search_results(raw_results: list[Mapping[str, Any]]) -> list[SearchResult]:
    """Normalize search results while preserving ordering and missing snippets."""

    enriched: list[SearchResult] = []
    for item in raw_results:
        result = normalize_search_result(item)
        if not result.url:
            continue
        enriched.append(result)
    return enriched
