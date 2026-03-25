from __future__ import annotations

import pytest
from urllib.error import HTTPError

from graneye.resolution import (
    ContextQuery,
    context_match_strength,
    detect_entity_type,
    detect_name_match_quality,
    rank_candidates,
    resolve_identity,
)
from graneye.search import SearchResult, enrich_search_results


@pytest.mark.parametrize(
    ("url", "expected_type"),
    [
        ("https://www.linkedin.com/in/jane-doe-123/", "person_profile"),
        ("https://example.com/directory/people", "directory"),
        ("https://example.com/company/acme", "company_page"),
        ("https://example.com/news/jane-doe-award", "article"),
    ],
)
def test_entity_type_detection(url: str, expected_type: str) -> None:
    result = SearchResult(title="Jane Doe", url=url, domain="example.com", snippet="")
    assert detect_entity_type(result) == expected_type


@pytest.mark.parametrize(
    ("query", "title", "snippet", "expected"),
    [
        ("José Álvarez", "José Álvarez - Profile", "", "full_match"),
        ("Gómez Martínez", "Martínez Gómez", "", "reordered_match"),
        ("Jane Doe", "J. Doe", "Senior engineer Jane", "partial_match"),
        ("Alice Brown", "Company team", "leadership page", "weak_match"),
    ],
)
def test_name_match_quality_multilingual(query: str, title: str, snippet: str, expected: str) -> None:
    result = SearchResult(title=title, url="https://example.com/in/user", domain="example.com", snippet=snippet)
    assert detect_name_match_quality(query, result) == expected


@pytest.mark.parametrize(
    ("raw_results", "expected_top_url"),
    [
        (
            [
                {
                    "title": "Jane Doe - People directory",
                    "url": "https://agg.example.com/directory/people?q=jane",
                    "snippet": "Find people records and phone numbers",
                },
                {
                    "title": "Jane Doe - Cybersecurity Engineer",
                    "url": "https://profiles.example.com/in/jane-doe",
                    "snippet": "Jane Doe is a cybersecurity engineer in Austin",
                },
            ],
            "https://profiles.example.com/in/jane-doe",
        ),
        (
            [
                {
                    "title": "Carlos Ruiz - Team",
                    "url": "https://corp.example.com/team/carlos-ruiz",
                    "snippet": "Engineering leader",
                },
                {
                    "title": "Carlos Ruiz",
                    "url": "https://www.linkedin.com/in/carlos-ruiz/",
                    "snippet": "Data scientist based in Madrid",
                },
            ],
            "https://www.linkedin.com/in/carlos-ruiz/",
        ),
    ],
)
def test_directory_and_aggregator_pages_do_not_outrank_profiles(raw_results: list[dict[str, str]], expected_top_url: str) -> None:
    results = enrich_search_results(raw_results)
    ranked = rank_candidates(results, "Jane Doe", ContextQuery())
    assert ranked[0].result.url == expected_top_url


@pytest.mark.parametrize(
    ("raw_results", "profession", "location", "expected_top"),
    [
        (
            [
                {
                    "title": "Alex Kim | Data Engineer",
                    "url": "https://example.com/in/alex-kim-data",
                    "snippet": "Alex Kim is a data engineer based in Seattle",
                },
                {
                    "title": "Alex Kim | Attorney",
                    "url": "https://example.com/in/alex-kim-legal",
                    "snippet": "Alex Kim is an attorney based in Boston",
                },
            ],
            "data engineer",
            "Seattle",
            "https://example.com/in/alex-kim-data",
        )
    ],
)
def test_ambiguous_identity_uses_context(
    raw_results: list[dict[str, str]],
    profession: str,
    location: str,
    expected_top: str,
) -> None:
    ranked = rank_candidates(
        enrich_search_results(raw_results),
        "Alex Kim",
        ContextQuery(profession=profession, location=location),
    )
    assert ranked[0].result.url == expected_top


@pytest.mark.parametrize(
    ("with_snippet", "without_snippet", "expected_top"),
    [
        (
            {
                "title": "Miguel Torres",
                "url": "https://example.com/in/miguel-torres",
                "snippet": "Miguel Torres is a machine learning engineer in Seattle",
            },
            {
                "title": "Miguel Torres",
                "url": "https://example.com/in/miguel-torres-alt",
            },
            "https://example.com/in/miguel-torres",
        )
    ],
)
def test_snippet_signal_improves_ranking(
    with_snippet: dict[str, str],
    without_snippet: dict[str, str],
    expected_top: str,
) -> None:
    ranked = rank_candidates(
        enrich_search_results([without_snippet, with_snippet]),
        "Miguel Torres",
        ContextQuery(),
    )
    assert ranked[0].result.url == expected_top


def test_exact_name_weak_context_does_not_outrank_context_aligned_candidate() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "John Smith",
                    "url": "https://profiles.example.com/in/john-smith",
                    "snippet": "Official profile page",
                },
                {
                    "title": "John Smith - Software Engineer",
                    "url": "https://uk-dev.example.org/in/john-smith-london",
                    "snippet": "Software engineer based in London building distributed systems",
                },
            ]
        ),
        "John Smith",
        ContextQuery(profession="Software Engineer", location="London"),
    )
    assert ranked[0].result.url == "https://uk-dev.example.org/in/john-smith-london"
    assert any("weak_context_exact_name_penalty" in reason for reason in ranked[1].reasons)


def test_context_match_strength_works_with_title_only_reordered_context() -> None:
    result = SearchResult(
        title="Satya Nadella - Chairman and CEO at Microsoft",
        url="https://www.microsoft.com/en-us/about/leadership/satya-nadella",
        domain="microsoft.com",
        snippet=None,
    )
    score, reasons = context_match_strength(result, ContextQuery(profession="Microsoft CEO"))
    assert score > 0.25
    assert any(reason.startswith("profession_token_overlap") for reason in reasons)


def test_common_name_ranking_prefers_context_match_over_plain_exact_match() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Carlos Pérez",
                    "url": "https://profiles.example.com/in/carlos-perez",
                    "snippet": "Professional profile",
                },
                {
                    "title": "Carlos Pérez - Cybersecurity Specialist",
                    "url": "https://security-spain.example.org/in/carlos-perez",
                    "snippet": "Cybersecurity consultant in Spain",
                },
            ]
        ),
        "Carlos Pérez",
        ContextQuery(profession="Cybersecurity", location="Spain"),
    )
    assert ranked[0].result.url == "https://security-spain.example.org/in/carlos-perez"


@pytest.mark.parametrize(
    ("html", "expected_role", "expected_org", "expected_location"),
    [
        (
            """
            <html><head><title>Jane Doe | Security Engineer</title>
            <meta name='description' content='Jane Doe is a security engineer at Acme Corp based in Austin'></head>
            <body><h1>Jane Doe</h1><h2>About</h2><p>Experienced engineer.</p></body></html>
            """,
            "engineer",
            "Acme Corp based in Austin",
            "Austin",
        )
    ],
)
def test_top_candidate_extraction(html: str, expected_role: str, expected_org: str, expected_location: str) -> None:
    results = enrich_search_results(
        [
            {
                "title": "Jane Doe - profile",
                "url": "https://example.com/in/jane-doe",
                "snippet": "Security engineer in Austin",
            }
        ]
    )

    output = resolve_identity("Jane Doe", results, fetcher=lambda _url: html)

    assert output is not None
    assert output.possible_role == expected_role
    assert output.possible_organization == expected_org
    assert output.possible_location == expected_location
    assert output.entity_type == "person_profile"
    assert output.resolution_path == "full_content"
    assert output.fetch_status == "ok"


def test_resolve_identity_falls_back_when_fetch_is_blocked() -> None:
    results = enrich_search_results(
        [
            {
                "title": "Jane Doe - LinkedIn",
                "url": "https://www.linkedin.com/in/jane-doe",
                "snippet": "Security engineer in Austin",
            }
        ]
    )

    def _blocked(_url: str) -> str:
        raise HTTPError(url=_url, code=999, msg="blocked", hdrs=None, fp=None)

    output = resolve_identity("Jane Doe", results, fetcher=_blocked)

    assert output is not None
    assert output.resolution_path == "fetch_blocked"
    assert output.fetch_status == "http_error:999"
    assert output.source_url == "https://www.linkedin.com/in/jane-doe"
