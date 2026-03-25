from __future__ import annotations

import pytest
from urllib.error import HTTPError

from graneye.resolution import (
    ContextQuery,
    assess_query_validity,
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
        ("https://example.com/company/acme", "official_bio"),
        ("https://example.com/news/jane-doe-award", "article"),
        ("https://university.example.edu/faculty/ana-ruiz-lopez", "academic_profile"),
        ("https://lawfirm.example.com/attorneys/francois-dupont", "official_profile"),
        ("https://corp.example.com/leadership/hans-muller", "official_profile"),
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
                "https://corp.example.com/team/carlos-ruiz",
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
        ContextQuery(role=profession, location=location),
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
        ContextQuery(role="Software Engineer", location="London"),
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
    score, reasons = context_match_strength(result, ContextQuery(role="Microsoft CEO"))
    assert score > 0.25
    assert any(reason.startswith("role_token_overlap") for reason in reasons)


def test_non_person_wiki_page_not_mislabeled_as_person_profile() -> None:
    result = SearchResult(
        title="A - Wikipedia",
        url="https://en.wikipedia.org/wiki/A",
        domain="wikipedia.org",
        snippet="Article about the letter A",
    )
    assert detect_entity_type(result) != "person_profile"


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
        ContextQuery(role="Cybersecurity", location="Spain"),
    )
    assert ranked[0].result.url == "https://security-spain.example.org/in/carlos-perez"


def test_context_aligned_company_page_can_beat_low_context_linkedin() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "John Smith | LinkedIn",
                    "url": "https://www.linkedin.com/in/john-smith",
                    "snippet": "View John Smith's professional profile.",
                },
                {
                    "title": "John Smith - Platform Engineering Director | ExampleAI",
                    "url": "https://exampleai.com/company/leadership/john-smith",
                    "snippet": "John Smith is Platform Engineering Director at ExampleAI in London.",
                },
            ]
        ),
        "John Smith",
        ContextQuery(role="Platform Engineering Director", location="London"),
    )
    assert ranked[0].result.url == "https://exampleai.com/company/leadership/john-smith"


def test_official_executive_page_outranks_contextual_article() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Satya Nadella comments on AI strategy",
                    "url": "https://news.example.com/article/satya-nadella-ai-strategy",
                    "snippet": "Contextual news headline about Microsoft CEO Satya Nadella.",
                },
                {
                    "title": "Satya Nadella - Chairman and CEO",
                    "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                    "snippet": "Official Microsoft leadership profile.",
                },
            ]
        ),
        "Satya Nadella",
        ContextQuery(role="ceo", organization="microsoft"),
    )
    assert ranked[0].entity_type in {"official_profile", "official_bio"}
    assert ranked[0].result.domain == "microsoft.com"


def test_low_confidence_when_only_weak_generic_results_exist() -> None:
    output = resolve_identity(
        "John Smith",
        enrich_search_results(
            [
                {
                    "title": "John Smith - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/John_Smith",
                    "snippet": "John Smith may refer to many people",
                }
            ]
        ),
    )
    assert output is not None
    assert output.ambiguity_detected is True
    assert output.confidence_label == "low"


def test_ambiguous_common_name_sets_multiple_plausible_reason() -> None:
    output = resolve_identity(
        "John Smith",
        enrich_search_results(
            [
                {
                    "title": "John Smith | Software Engineer",
                    "url": "https://londondev.example.com/in/john-smith",
                    "snippet": "Software engineer in London",
                },
                {
                    "title": "John Smith | Backend Engineer",
                    "url": "https://ukcoder.example.com/in/john-smith",
                    "snippet": "Backend engineer based in London",
                },
            ]
        ),
        role="Software Engineer",
        location="London",
    )
    assert output is not None
    assert output.ambiguity_detected is True
    assert output.ambiguity_reason == "multiple_plausible_candidates"


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


def test_creator_profile_outweighs_generic_article_for_creator_context() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Ibai Llanos wins streaming awards",
                    "url": "https://news.example.com/article/ibai-awards",
                    "snippet": "Public figure recognized in Spain.",
                },
                {
                    "title": "Ibai Llanos - YouTube",
                    "url": "https://www.youtube.com/@Ibai",
                    "snippet": "Streamer and creator from Spain.",
                },
            ]
        ),
        "Ibai Llanos",
        ContextQuery(role="streamer", media_platform="youtube", location="Spain"),
    )
    assert ranked[0].entity_type == "creator_profile"
    assert ranked[0].result.url == "https://www.youtube.com/@Ibai"


def test_academic_profile_outweighs_news_when_context_is_faculty() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "María García appears in conference news",
                    "url": "https://press.example.com/news/maria-garcia-conference",
                    "snippet": "Professor from Madrid discusses education.",
                },
                {
                    "title": "María García - Faculty Profile",
                    "url": "https://university.example.edu/faculty/maria-garcia",
                    "snippet": "Professor at University of Madrid.",
                },
            ]
        ),
        "María García",
        ContextQuery(role="professor", institutional_hint="university", location="Madrid"),
    )
    assert ranked[0].entity_type == "academic_profile"


@pytest.mark.parametrize(
    ("query", "expected_status"),
    [("a", "too_short"), ("123456", "numeric_or_garbage"), ("profile", "too_generic"), ("John Smith", "valid")],
)
def test_query_validity_assessment_handles_invalid_and_valid_inputs(query: str, expected_status: str) -> None:
    assert assess_query_validity(query).status == expected_status


def test_low_authority_bio_page_gets_penalized_against_official_profile() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Ada Lovelace age net worth family biography facts",
                    "url": "https://celeb-bio-now.example.com/ada-lovelace-biography",
                    "snippet": "Age, net worth, relationship, family and career facts.",
                },
                {
                    "title": "Ada Lovelace - Encyclopaedia Britannica",
                    "url": "https://www.britannica.com/biography/Ada-Lovelace",
                    "snippet": "English mathematician and pioneer in computing.",
                },
            ]
        ),
        "Ada Lovelace",
        ContextQuery(),
    )
    assert ranked[0].result.domain == "britannica.com"
    assert ranked[1].authority_tier == "low_authority_bio_seo"
    assert ranked[1].seo_penalty > 0.1


def test_wikipedia_stays_low_confidence_when_only_weak_evidence_exists() -> None:
    output = resolve_identity(
        "Carlos Pérez",
        enrich_search_results(
            [
                {
                    "title": "Carlos Pérez - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Carlos_P%C3%A9rez",
                    "snippet": "Carlos Pérez may refer to multiple people.",
                }
            ]
        ),
    )
    assert output is not None
    assert output.confidence_label == "low"
    assert "query_validity=valid" in output.explanation


def test_invalid_numeric_query_forces_low_confidence() -> None:
    output = resolve_identity(
        "123456",
        enrich_search_results(
            [
                {
                    "title": "123456 - Profile",
                    "url": "https://example.com/profile/123456",
                    "snippet": "Generic profile record",
                }
            ]
        ),
    )
    assert output is not None
    assert output.confidence_label == "low"
    assert output.ambiguity_reason == "invalid_query:numeric_or_garbage"


def test_invalid_short_query_applies_explicit_score_cap() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "A profile",
                    "url": "https://example.com/in/a",
                    "snippet": "Generic profile page",
                }
            ]
        ),
        "a",
        ContextQuery(),
    )
    assert ranked[0].score <= 0.42
    assert ranked[0].score_cap_applied == 0.42
    assert any("query_invalid_score_cap:too_short:0.42" in reason for reason in ranked[0].reasons)


def test_encyclopedic_source_has_reduced_seo_penalty() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Ada Lovelace - Encyclopaedia Britannica biography facts",
                    "url": "https://www.britannica.com/biography/Ada-Lovelace",
                    "snippet": "Biography and facts about Ada Lovelace.",
                },
                {
                    "title": "Ada Lovelace age net worth family biography facts",
                    "url": "https://celeb-bio-now.example.com/ada-lovelace-biography",
                    "snippet": "Age, net worth, relationship, family and career facts.",
                },
            ]
        ),
        "Ada Lovelace",
        ContextQuery(),
    )
    brit = next(item for item in ranked if item.result.domain == "britannica.com")
    seo = next(item for item in ranked if item.result.domain == "celeb-bio-now.example.com")
    assert brit.seo_penalty < 0.05
    assert seo.seo_penalty > brit.seo_penalty


def test_common_name_without_context_remains_low_confidence() -> None:
    output = resolve_identity(
        "John Smith",
        enrich_search_results(
            [
                {
                    "title": "John Smith | LinkedIn",
                    "url": "https://www.linkedin.com/in/john-smith",
                    "snippet": "View John Smith's profile",
                },
                {
                    "title": "John Smith - Software Engineer",
                    "url": "https://portfolio.example.com/john-smith",
                    "snippet": "Software engineer portfolio",
                },
            ]
        ),
    )
    assert output is not None
    assert output.confidence_label == "low"
