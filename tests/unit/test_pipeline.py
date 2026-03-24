from __future__ import annotations

from graneye.pipeline import resolve_query


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
