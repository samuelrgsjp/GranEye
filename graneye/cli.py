from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import urlopen

from .pipeline import resolve_query
from .resolution import ResolutionOutput
from .search import SearchResult


@dataclass(slots=True, frozen=True)
class CLIArgs:
    target_name: str
    target_context: str | None


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture_link = False
        self._capture_snippet = False
        self._current: dict[str, str] | None = None
        self.results: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {k: (v or "") for k, v in attrs}
        klass = attrs_map.get("class", "")

        if tag == "a" and "result__a" in klass:
            href = attrs_map.get("href", "").strip()
            resolved_url = _resolve_ddg_redirect(href)
            self._current = {"title": "", "url": resolved_url, "snippet": ""}
            self._capture_link = True

        if tag in {"a", "div"} and "result__snippet" in klass and self._current:
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._capture_link = False
            self._capture_snippet = False
        if tag == "div":
            self._capture_snippet = False

        if tag == "article" and self._current:
            if self._current.get("url"):
                self.results.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if not self._current:
            return
        text = " ".join(data.split())
        if not text:
            return

        if self._capture_link:
            self._current["title"] = f"{self._current['title']} {text}".strip()
        elif self._capture_snippet:
            self._current["snippet"] = f"{self._current['snippet']} {text}".strip()


def _resolve_ddg_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path != "/l/":
        return url

    query = parse_qs(parsed.query)
    uddg_values = query.get("uddg")
    if not uddg_values:
        return url
    return uddg_values[0]


def _search_duckduckgo_html(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    encoded = quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    with urlopen(url, timeout=10) as response:
        html = response.read(600_000).decode("utf-8", errors="ignore")

    parser = _DuckDuckGoParser()
    parser.feed(html)
    return parser.results[:max_results]


def _search_duckduckgo_instant_answer(query: str, *, max_results: int = 10) -> list[dict[str, str]]:
    encoded = quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
    with urlopen(url, timeout=10) as response:
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
        url = str(item.get("FirstURL", "")).strip()
        text = str(item.get("Text", "")).strip()
        if not url:
            return
        title = text.split(" - ")[0].strip() if text else ""
        normalized.append({"title": title, "url": url, "snippet": text})

    for topic in payload.get("RelatedTopics", []):
        if isinstance(topic, dict) and "FirstURL" in topic:
            append_topic(topic)
        elif isinstance(topic, dict):
            for child in topic.get("Topics", []):
                if isinstance(child, dict):
                    append_topic(child)

    return normalized[:max_results]


def _parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="graneye",
        description="Resolve a likely public profile candidate for a person name.",
    )
    parser.add_argument("target_name", help="Target person name to resolve")
    parser.add_argument(
        "target_context",
        nargs="?",
        default=None,
        help="Optional disambiguation context (for example: 'Lawyer Barcelona')",
    )

    namespace = parser.parse_args(argv)
    return CLIArgs(target_name=namespace.target_name, target_context=namespace.target_context)


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _render_output(target_name: str, target_context: str | None, output: ResolutionOutput, top: SearchResult) -> str:
    lines = [
        f"Target name: {target_name}",
        f"Target context: {target_context}" if target_context else "Target context: (none)",
        f"Top candidate: {output.normalized_candidate_name}",
        f"Source URL: {output.source_url}",
        f"Display title: {top.title or '(not available)'}",
        f"Score: {output.final_score:.3f}",
        f"Same-person probability: {output.same_person_probability:.3f}",
        f"Context match probability: {output.context_match_probability:.3f}",
        f"Decision reason: {output.explanation}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if _is_blank(args.target_name):
        print("Error: target_name must not be empty.", file=sys.stderr)
        return 2

    context = args.target_context.strip() if args.target_context else None

    try:
        output, ranked = resolve_query(
            args.target_name.strip(),
            context=context,
            html_search=_search_duckduckgo_html,
            instant_search=_search_duckduckgo_instant_answer,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error: failed to execute pipeline: {exc}", file=sys.stderr)
        return 1

    if output is None or not ranked:
        print(f"No candidates found for '{args.target_name.strip()}'.")
        return 3

    print(_render_output(args.target_name.strip(), context, output, ranked[0].result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
