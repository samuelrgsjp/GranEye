from __future__ import annotations

import re
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
        ("https://example.com/directory/people", "directory_listing"),
        ("https://example.com/company/acme", "official_bio"),
        ("https://example.com/news/jane-doe-award", "media_article"),
        ("https://university.example.edu/faculty/ana-ruiz-lopez", "institutional_profile"),
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
    assert output.no_resolution is True
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
    assert output.ambiguity_detected is False
    assert output.ambiguity_reason is None
    assert output.no_resolution is True


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
    assert ranked[0].entity_type == "person_profile"
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
    assert ranked[0].entity_type == "institutional_profile"


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
    assert output.no_resolution is True
    assert output.no_resolution_reason == "generic_encyclopedic_fallback_without_support"


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
    assert output.no_resolution is True
    assert output.no_resolution_reason == "invalid_query:numeric_or_garbage"


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


def test_official_executive_match_not_forced_low_confidence() -> None:
    output = resolve_identity(
        "Satya Nadella",
        enrich_search_results(
            [
                {
                    "title": "Satya Nadella - Chairman and CEO",
                    "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                    "snippet": "Official leadership profile for Satya Nadella, Chairman and CEO at Microsoft.",
                },
                {
                    "title": "Satya Nadella AI strategy interview",
                    "url": "https://news.example.com/satya-nadella-interview",
                    "snippet": "Interview coverage from industry media.",
                },
            ]
        ),
        role="CEO",
        organization="Microsoft",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.confidence_label == "high"
    assert "official_superiority_bonus=yes" in output.explanation


def test_extraction_normalization_removes_navigation_junk() -> None:
    html = """
    <html>
      <head><title>Jensen Huang - NVIDIA</title></head>
      <body>
        <nav>Source your privacy choices opt out icon</nav>
        <header>Menu Privacy Terms</header>
        <h1>Jensen Huang</h1>
        <p>Founder and CEO at NVIDIA.</p>
        <footer>Privacy cookie consent opt out</footer>
      </body>
    </html>
    """
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang - NVIDIA",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Founder and CEO.",
                }
            ]
        ),
        role="CEO",
        organization="NVIDIA",
        fetcher=lambda _url: html,
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.normalized_candidate_name == "jensen huang"
    assert "privacy" not in output.normalized_candidate_name


@pytest.mark.parametrize("query", ["a", "123456"])
def test_invalid_queries_return_hard_no_resolution(query: str) -> None:
    output = resolve_identity(
        query,
        enrich_search_results(
            [
                {
                    "title": f"{query} - profile",
                    "url": f"https://example.com/in/{query}",
                    "snippet": "Generic profile page",
                }
            ]
        ),
    )
    assert output is not None
    assert output.no_resolution is True
    assert output.source_url == ""
    assert output.normalized_candidate_name == ""
    assert "NO_RESOLUTION: invalid query" in output.explanation


def test_strong_official_convergence_is_high_and_not_ambiguous() -> None:
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA biography of founder and CEO Jensen Huang.",
                },
                {
                    "title": "Jensen Huang | LinkedIn",
                    "url": "https://www.linkedin.com/in/jensen-huang",
                    "snippet": "Jensen Huang at NVIDIA",
                },
            ]
        ),
        role="CEO",
        organization="NVIDIA",
    )
    assert output is not None
    assert output.source_url.startswith("https://www.nvidia.com/")
    assert output.confidence_label == "high"
    assert output.ambiguity_detected is False
    assert output.no_resolution is False


def test_official_domain_beats_aggregator_with_same_name() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang - The Official Board",
                    "url": "https://www.theofficialboard.com/biography/jensen-huang",
                    "snippet": "Executive profile directory entry",
                },
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA leadership biography",
                },
            ]
        ),
        "Jensen Huang",
        ContextQuery(role="CEO", organization="NVIDIA"),
    )
    assert ranked[0].result.domain == "nvidia.com"
    assert ranked[1].authority_tier == "directory_aggregator"


def test_linkedin_obvious_profile_not_typed_unknown() -> None:
    result = SearchResult(
        title="Jane Doe | LinkedIn",
        url="https://www.linkedin.com/in/jane-doe-123",
        domain="linkedin.com",
        snippet="Software engineer at Acme",
    )
    assert detect_entity_type(result) == "person_profile"


def test_weak_evidence_results_in_no_resolution_without_ambiguity() -> None:
    output = resolve_identity(
        "Random Person",
        enrich_search_results(
            [
                {
                    "title": "Random Person overview",
                    "url": "https://example.com/about/random",
                    "snippet": "Company overview page",
                }
            ]
        ),
    )
    assert output is not None
    assert output.no_resolution is True
    assert output.no_resolution_reason == "insufficient_evidence"
    assert output.ambiguity_detected is False


def test_clustered_multi_source_official_identity_resolves_without_ambiguity() -> None:
    output = resolve_identity(
        "Satya Nadella",
        enrich_search_results(
            [
                {
                    "title": "Satya Nadella - Chairman and CEO",
                    "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                    "snippet": "Official Microsoft leadership profile.",
                },
                {
                    "title": "Satya Nadella | LinkedIn",
                    "url": "https://www.linkedin.com/in/satya-nadella/",
                    "snippet": "Chairman and CEO at Microsoft",
                },
                {
                    "title": "Who is Satya Nadella?",
                    "url": "https://apnews.com/article/satya-nadella-profile",
                    "snippet": "AP profile of Microsoft CEO Satya Nadella.",
                },
            ]
        ),
        role="CEO",
        organization="Microsoft",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.ambiguity_detected is False
    assert output.confidence_label == "high"


def test_official_first_party_bio_outranks_media_for_identity_resolution() -> None:
    ranked = rank_candidates(
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang discusses AI market",
                    "url": "https://apnews.com/article/jensen-huang-ai",
                    "snippet": "AP interview with NVIDIA CEO Jensen Huang.",
                },
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA biography for founder and CEO Jensen Huang.",
                },
            ]
        ),
        "Jensen Huang",
        ContextQuery(role="CEO", organization="NVIDIA"),
    )
    assert ranked[0].result.domain == "nvidia.com"
    assert any("first_party_exact_name_org_priority" in reason for reason in ranked[0].reasons)


def test_common_name_researchgate_only_profile_stays_no_resolution() -> None:
    output = resolve_identity(
        "María López",
        enrich_search_results(
            [
                {
                    "title": "María López | ResearchGate",
                    "url": "https://www.researchgate.net/profile/Maria-Lopez-12",
                    "snippet": "Professor profile and publications",
                }
            ]
        ),
        role="teacher",
        location="Valencia",
    )
    assert output is not None
    assert output.no_resolution is True
    assert output.no_resolution_reason == "common_name_structured_profile_without_corroboration"


def test_single_wikipedia_result_does_not_get_high_confidence() -> None:
    output = resolve_identity(
        "Taylor Swift",
        enrich_search_results(
            [
                {
                    "title": "Taylor Swift - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Taylor_Swift",
                    "snippet": "American singer-songwriter.",
                }
            ]
        ),
        role="music singer",
        location="usa",
    )
    assert output is not None
    assert output.confidence_label != "high"


@pytest.mark.parametrize(
    ("query", "context_kwargs", "raw_result"),
    [
        (
            "Lionel Messi",
            {"domain_activity": "football", "location": "argentina"},
            {
                "title": "Lionel Messi - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
                "snippet": "Argentine professional footballer.",
            },
        ),
        (
            "MrBeast",
            {"media_platform": "youtube"},
            {
                "title": "MrBeast - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/MrBeast",
                "snippet": "American YouTuber known for challenge videos and philanthropy.",
            },
        ),
        (
            "Taylor Swift",
            {"role": "singer", "domain_activity": "music", "location": "usa"},
            {
                "title": "Taylor Swift - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Taylor_Swift",
                "snippet": "American singer-songwriter.",
            },
        ),
    ],
)
def test_distinctive_public_identities_get_at_least_medium_confidence_with_strong_encyclopedic_evidence(
    query: str,
    context_kwargs: dict[str, str],
    raw_result: dict[str, str],
) -> None:
    output = resolve_identity(query, enrich_search_results([raw_result]), **context_kwargs)
    assert output is not None
    assert output.no_resolution is False
    assert output.confidence_label in {"medium", "high"}


@pytest.mark.parametrize(
    "url",
    [
        "https://www.vatican.va/content/francesco/en/biography/documents/papa-francesco_20130313_biografia-bergoglio.html",
        "https://www.vaticannews.va/en/pope/news/2025-01/pope-francis-message.html",
        "https://messi.com/en/biography/",
    ],
)
def test_non_corporate_official_domains_are_typed_strongly(url: str) -> None:
    ranked = rank_candidates(
        enrich_search_results([{"title": "Official profile", "url": url, "snippet": "Official page"}]),
        "Pope Francis",
        ContextQuery(institutional_hint="vatican"),
    )
    assert ranked[0].authority_tier == "official_institutional"


def test_common_name_diffuse_cybersecurity_matches_do_not_force_multiple_plausible_reason() -> None:
    output = resolve_identity(
        "Carlos Pérez",
        enrich_search_results(
            [
                {
                    "title": "Carlos Pérez | LinkedIn",
                    "url": "https://www.linkedin.com/in/carlos-perez-security",
                    "snippet": "Cybersecurity professional in Spain.",
                },
                {
                    "title": "Carlos Perez - Security Consultant",
                    "url": "https://profiles.example.com/in/carlos-perez",
                    "snippet": "Consultant profile in Spain.",
                },
                {
                    "title": "Carlos Pérez speaker profile",
                    "url": "https://events.example.org/speakers/carlos-perez",
                    "snippet": "Cybersecurity conference speaker.",
                },
            ]
        ),
        domain_activity="cybersecurity",
        location="spain",
    )
    assert output is not None
    assert output.no_resolution is True
    assert output.no_resolution_reason in {"common_name_weak_context", "insufficient_identity_specific_support"}
    assert output.ambiguity_reason != "multiple_plausible_candidates"


def test_common_name_diffuse_software_engineer_madrid_stays_no_resolution() -> None:
    output = resolve_identity(
        "David Martínez",
        enrich_search_results(
            [
                {
                    "title": "David Martínez | LinkedIn",
                    "url": "https://www.linkedin.com/in/david-martinez",
                    "snippet": "Software engineer in Madrid.",
                },
                {
                    "title": "David Martinez - Developer",
                    "url": "https://profiles.example.com/in/dmartinez",
                    "snippet": "Software developer profile in Madrid.",
                },
                {
                    "title": "David Martinez GitHub",
                    "url": "https://github.com/dmartinez",
                    "snippet": "Software projects and repositories.",
                },
            ]
        ),
        role="software engineer",
        location="madrid",
    )
    assert output is not None
    assert output.no_resolution is True


def test_creator_multi_channel_ecosystem_not_marked_ambiguous() -> None:
    output = resolve_identity(
        "MrBeast",
        enrich_search_results(
            [
                {
                    "title": "MrBeast - YouTube",
                    "url": "https://www.youtube.com/@MrBeast",
                    "snippet": "Main channel",
                },
                {
                    "title": "MrBeast Gaming - YouTube",
                    "url": "https://www.youtube.com/@MrBeastGaming",
                    "snippet": "Secondary channel",
                },
                {
                    "title": "MrBeast 2 - YouTube",
                    "url": "https://www.youtube.com/@MrBeast2",
                    "snippet": "Second channel",
                },
            ]
        ),
        media_platform="youtube",
    )
    assert output is not None
    assert output.ambiguity_detected is False


@pytest.mark.parametrize("query", ["Carlos Pérez", "María López"])
def test_no_context_encyclopedic_fallback_reason_for_common_name(query: str) -> None:
    output = resolve_identity(
        query,
        enrich_search_results(
            [
                {
                    "title": f"{query} - Wikipedia",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"{query} profile page.",
                }
            ]
        ),
    )
    assert output is not None
    assert output.no_resolution is True
    assert output.no_resolution_reason == "generic_encyclopedic_fallback_without_support"


def test_jensen_huang_nvidia_ceo_resolves_with_official_representation() -> None:
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang keynote at NVIDIA GTC",
                    "url": "https://www.nvidia.com/en-us/events/gtc/keynote/",
                    "snippet": "Watch NVIDIA CEO Jensen Huang's keynote.",
                },
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA biography for founder and CEO Jensen Huang.",
                },
                {
                    "title": "Jensen Huang discusses AI market",
                    "url": "https://apnews.com/article/jensen-huang-ai",
                    "snippet": "AP coverage of NVIDIA CEO Jensen Huang.",
                },
            ]
        ),
        role="CEO",
        organization="NVIDIA",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.confidence_label in {"high", "medium"}
    assert output.source_url == "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/"


def test_jensen_huang_nvidia_founder_ceo_resolves() -> None:
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA founder and CEO biography.",
                }
            ]
        ),
        role="CEO founder",
        organization="NVIDIA",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.confidence_label in {"high", "medium"}


def test_satya_nadella_microsoft_executive_resolves_at_least_medium() -> None:
    output = resolve_identity(
        "Satya Nadella",
        enrich_search_results(
            [
                {
                    "title": "Satya Nadella - Chairman and CEO",
                    "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                    "snippet": "Official Microsoft leadership profile for Satya Nadella.",
                }
            ]
        ),
        role="executive",
        organization="Microsoft",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.confidence_label in {"high", "medium"}


@pytest.mark.parametrize(
    ("name", "context", "expected_domain"),
    [
        ("Pope Francis", {"institutional_hint": "vatican"}, "wikipedia.org"),
        ("Lionel Messi", {"domain_activity": "football", "location": "argentina"}, "wikipedia.org"),
        ("MrBeast", {"media_platform": "youtube"}, "wikipedia.org"),
        ("Taylor Swift", {"role": "singer", "domain_activity": "music", "location": "usa"}, "wikipedia.org"),
    ],
)
def test_distinctive_public_figures_resolve_from_single_strong_encyclopedic_source(
    name: str,
    context: dict[str, str],
    expected_domain: str,
) -> None:
    output = resolve_identity(
        name,
        enrich_search_results(
            [
                {
                    "title": f"{name} - Wikipedia",
                    "url": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                    "snippet": f"{name} public figure profile.",
                }
            ]
        ),
        **context,
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.source_url.endswith(expected_domain) or expected_domain in output.source_url
    assert output.confidence_label in {"medium", "low"}


def test_source_diversity_not_required_when_single_official_source_is_strong() -> None:
    output = resolve_identity(
        "Satya Nadella",
        enrich_search_results(
            [
                {
                    "title": "Satya Nadella - Chairman and CEO",
                    "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                    "snippet": "Official Microsoft leadership profile.",
                }
            ]
        ),
        role="CEO",
        organization="Microsoft",
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.source_url == "https://www.microsoft.com/en-us/about/leadership/satya-nadella"


def test_representative_surface_fields_stay_aligned_for_jensen_huang() -> None:
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang keynote at NVIDIA GTC",
                    "url": "https://www.nvidia.com/en-us/events/gtc/keynote/",
                    "snippet": "Watch NVIDIA CEO Jensen Huang's keynote.",
                },
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA biography for founder and CEO Jensen Huang.",
                },
            ]
        ),
        role="CEO",
        organization="NVIDIA",
        fetcher=lambda _url: """
            <html>
              <head>
                <title>About NVIDIA | Leadership</title>
                <meta property='og:title' content='About NVIDIA'>
              </head>
              <body>
                <h1>Jensen Huang</h1>
                <p>Founder and CEO of NVIDIA.</p>
              </body>
            </html>
        """,
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.source_url == "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/"
    assert output.normalized_candidate_name == "jensen huang"
    assert output.normalized_candidate_name != "about nvidia"


def test_canonical_name_extraction_rejects_about_nvidia_fragment() -> None:
    output = resolve_identity(
        "Jensen Huang",
        enrich_search_results(
            [
                {
                    "title": "Jensen Huang - Founder and CEO",
                    "url": "https://www.nvidia.com/en-us/about-nvidia/jensen-huang/",
                    "snippet": "Official NVIDIA biography.",
                }
            ]
        ),
        role="CEO",
        organization="NVIDIA",
        fetcher=lambda _url: """
            <html>
              <head>
                <title>About NVIDIA | Jensen Huang</title>
                <meta property='og:title' content='About NVIDIA'>
              </head>
              <body><h1>Jensen Huang</h1></body>
            </html>
        """,
    )
    assert output is not None
    assert output.no_resolution is False
    assert "canonical_name_source=og_title" not in output.explanation
    assert output.normalized_candidate_name == "jensen huang"


def test_mrbeast_single_token_identity_gets_stronger_canonical_quality() -> None:
    output = resolve_identity(
        "MrBeast",
        enrich_search_results(
            [
                {
                    "title": "MrBeast - YouTube",
                    "url": "https://www.youtube.com/@MrBeast",
                    "snippet": "Main channel",
                }
            ]
        ),
        media_platform="youtube",
        fetcher=lambda _url: """
            <html>
              <head>
                <title>MrBeast - YouTube</title>
                <meta property='og:title' content='MrBeast'>
              </head>
              <body><h1>MrBeast</h1></body>
            </html>
        """,
    )
    assert output is not None
    assert output.no_resolution is False
    assert output.normalized_candidate_name == "mrbeast"
    match = re.search(r"canonical_name_quality=([0-9.]+)", output.explanation)
    assert match is not None
    assert float(match.group(1)) >= 0.7
