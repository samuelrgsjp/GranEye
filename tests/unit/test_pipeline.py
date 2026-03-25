from __future__ import annotations

from graneye.pipeline import _parse_context, _query_variants, resolve_query, resolve_query_with_debug


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


def test_resolve_query_retries_with_name_only_when_context_query_is_empty() -> None:
    seen_queries: list[str] = []

    def _html_search(query: str) -> list[dict[str, str]]:
        seen_queries.append(query)
        if "Microsoft CEO" in query:
            return []
        return [{"title": "Satya Nadella - Microsoft", "url": "https://example.com/satya", "snippet": "CEO at Microsoft"}]

    output, ranked = resolve_query(
        "Satya Nadella",
        context="Microsoft CEO",
        html_search=_html_search,
        instant_search=lambda _q: [],
    )
    assert output is not None
    assert ranked
    assert seen_queries[0] == "Satya Nadella Microsoft CEO"
    assert "\"Satya Nadella\" Microsoft CEO" in seen_queries
    assert "Satya Nadella" in seen_queries


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
    assert diagnostics.query_attempts[0] == "Satya Nadella Microsoft CEO"
    assert diagnostics.query_validity == "valid"
    assert any(decision.reason == "search_engine_or_internal" for decision in diagnostics.filter_decisions)
    assert diagnostics.ranked_candidates
    assert diagnostics.source_diversity_count >= 1


def test_resolve_query_with_debug_high_signal_instant_fallback_has_candidates() -> None:
    instant_results = [
        {
            "title": "Jensen Huang - NVIDIA Leadership",
            "url": "https://www.nvidia.com/en-us/about-nvidia/leadership/",
            "snippet": "Founder, President and CEO of NVIDIA",
        }
    ]
    output, ranked, diagnostics = resolve_query_with_debug(
        "Jensen Huang",
        context="NVIDIA CEO",
        html_search=lambda _q: [],
        instant_search=lambda _q: instant_results,
    )
    assert output is not None
    assert diagnostics.raw_results_count == 1
    assert diagnostics.filtered_results_count == 1
    assert ranked[0].context_strength > 0


def test_resolve_query_with_debug_reports_ambiguity_when_candidates_are_close() -> None:
    html_results = [
        {"title": "John Smith - Engineer", "url": "https://a.example.com/john-smith", "snippet": "Engineer in London"},
        {"title": "John Smith - Developer", "url": "https://b.example.com/john-smith", "snippet": "Software engineer London"},
    ]
    output, _ranked, diagnostics = resolve_query_with_debug(
        "John Smith",
        context="Software Engineer London",
        html_search=lambda _q: html_results,
        instant_search=lambda _q: [],
    )
    assert output is not None
    assert output.ambiguity_detected is True
    assert diagnostics.ambiguity_triggered is True


def test_parse_context_supports_public_creator_hints() -> None:
    parsed = _parse_context("Streamer Spain YouTube")
    assert parsed.role == "streamer"
    assert parsed.media_platform == "youtube"
    assert parsed.location == "Spain"
    assert parsed.domain_activity is None
    assert "streamer" in parsed.generic_terms


def test_parse_context_disambiguates_role_org_location_activity() -> None:
    microsoft = _parse_context("Microsoft CEO")
    assert microsoft.role == "ceo"
    assert microsoft.organization == "Microsoft"
    assert microsoft.location is None

    cyber = _parse_context("Cybersecurity Spain")
    assert cyber.role is None
    assert cyber.domain_activity == "cybersecurity"
    assert cyber.location == "Spain"

    engineer = _parse_context("Software Engineer Madrid")
    assert engineer.role == "software engineer"
    assert engineer.location == "Madrid"

    founder = _parse_context("NVIDIA founder CEO")
    assert founder.organization == "NVIDIA"
    assert founder.role is not None
    assert "ceo" in founder.role

    messi = _parse_context("football argentina")
    assert messi.domain_activity == "football"
    assert messi.location == "argentina"


def test_query_variants_expand_beyond_professional_context() -> None:
    variants = _query_variants("Penélope Cruz", "Actress")
    assert "Penélope Cruz Actress official" in variants
    assert "Penélope Cruz Actress wikipedia" in variants
    assert "\"Penélope Cruz\" Actress official bio" in variants


def test_resolve_query_with_debug_marks_invalid_short_query() -> None:
    output, ranked, diagnostics = resolve_query_with_debug(
        "a",
        context=None,
        html_search=lambda _q: [{"title": "A profile", "url": "https://example.com/a-profile", "snippet": "profile"}],
        instant_search=lambda _q: [],
    )
    assert output is not None
    assert ranked
    assert diagnostics.query_validity == "too_short"
    assert output.confidence_label == "low"
