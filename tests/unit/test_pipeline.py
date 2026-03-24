from __future__ import annotations

from graneye.pipeline import resolve_query, resolve_query_with_debug


def test_resolve_query_merges_html_and_instant_results_without_duplicates() -> None:
    html_results = [
        {"title": "Jane Doe - Profile", "url": "https://example.com/in/jane-doe", "snippet": "Engineer"},
    ]
    instant_results = [
        {"title": "Jane Doe", "url": "https://example.com/in/jane-doe", "snippet": "Duplicate"},
        {"title": "Jane Doe Wiki", "url": "https://en.wikipedia.org/wiki/Jane_Doe", "snippet": "Article"},
    ]

    output, ranked = resolve_query(
        "Jane Doe",
        context="Engineer Seattle",
        html_search=lambda _q: html_results,
        instant_search=lambda _q: instant_results,
    )

    assert output is not None
    urls = [item.result.url for item in ranked]
    assert urls.count("https://example.com/in/jane-doe") == 1
    assert "https://en.wikipedia.org/wiki/Jane_Doe" in urls


def test_resolve_query_returns_no_output_when_all_searches_empty() -> None:
    output, ranked = resolve_query(
        "Jane Doe",
        context=None,
        html_search=lambda _q: [],
        instant_search=lambda _q: [],
    )

    assert output is None
    assert ranked == []


def test_resolve_query_with_debug_reports_counts_and_filter_decisions() -> None:
    html_results = [
        {"title": "Satya Nadella - Microsoft", "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella", "snippet": "Chairman and CEO"},
        {"title": "DuckDuckGo privacy", "url": "https://duckduckgo.com/privacy", "snippet": "privacy page"},
    ]

    output, ranked, diagnostics = resolve_query_with_debug(
        "Satya Nadella",
        context="Microsoft CEO",
        html_search=lambda _q: html_results,
        instant_search=lambda _q: [],
    )

    assert output is not None
    assert ranked
    assert diagnostics.raw_results_count == 2
    assert diagnostics.normalized_results_count == 2
    assert diagnostics.filtered_results_count == 1
    assert diagnostics.ranked_candidates_count == len(ranked)
    assert any(decision.reason == "search_engine_or_internal" for decision in diagnostics.filter_decisions)
