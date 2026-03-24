from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse


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


def normalize_search_result(payload: Mapping[str, Any]) -> SearchResult:
    """Normalize a raw search payload into a stable SearchResult object."""

    title = _to_text(payload.get("title"))
    snippet_raw = _to_text(payload.get("snippet") or payload.get("description"))
    url = _to_text(payload.get("url") or payload.get("link"))

    parsed = urlparse(url)
    domain = parsed.netloc.casefold().lstrip("www.") if parsed.netloc else ""

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
