from __future__ import annotations

import io
import json
from urllib.parse import quote_plus

from graneye import search


SAMPLE_DDG_HTML = """
<html>
  <body>
    <article class="result result--web">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fes.linkedin.com%2Fin%2Flaura-gomez-martinez">
          Laura Gómez Martínez - LinkedIn
        </a>
      </h2>
      <div class="result__snippet">Lawyer in Barcelona at Lex Group.</div>
    </article>
    <article class="result result--web">
      <h2><a class="result__a" href="https://example.org/profile">Laura Gómez bio</a></h2>
      <a class="result__snippet">Public profile page in Spain</a>
    </article>
  </body>
</html>
"""

SAMPLE_DDG_HTML_FALLBACK = """
<html>
  <body>
    <div class="links_main links_deep result__body">
      <a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fin%2Fjane-doe">Jane Doe profile</a>
    </div>
  </body>
</html>
"""

SAMPLE_DDG_HTML_INTERNAL = """
<html>
  <body>
    <article class="result result--web">
      <h2 class="result__title">
        <a class="result__a" href="https://duckduckgo.com/privacy">here</a>
      </h2>
      <div class="result__snippet">DuckDuckGo protection privacy peace of mind</div>
    </article>
    <article class="result result--web">
      <h2 class="result__title">
        <a class="result__a" href="https://example.net/in/laura-gomez">Laura Gómez Martínez - Bio</a>
      </h2>
      <div class="result__snippet">Lawyer in Barcelona.</div>
    </article>
  </body>
</html>
"""


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_parse_duckduckgo_html_results_extracts_title_url_and_snippet() -> None:
    results = search.parse_duckduckgo_html_results(SAMPLE_DDG_HTML, max_results=10)

    assert len(results) == 2
    assert results[0]["title"] == "Laura Gómez Martínez - LinkedIn"
    assert results[0]["url"] == "https://es.linkedin.com/in/laura-gomez-martinez"
    assert "Lawyer in Barcelona" in results[0]["snippet"]
    assert results[1]["url"] == "https://example.org/profile"


def test_search_duckduckgo_html_uses_html_endpoint_and_headers(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_urlopen(request, timeout: int = 10):
        captured["url"] = request.full_url
        captured["ua"] = request.headers.get("User-agent", "")
        captured["accept"] = request.headers.get("Accept", "")
        assert timeout == 10
        return _FakeResponse(SAMPLE_DDG_HTML.encode("utf-8"))

    monkeypatch.setattr(search, "urlopen", _fake_urlopen)

    results = search.search_duckduckgo_html("Satya Nadella Microsoft CEO", max_results=1)

    assert results
    assert captured["url"] == f"https://html.duckduckgo.com/html/?q={quote_plus('Satya Nadella Microsoft CEO')}"
    assert "Mozilla/5.0" in captured["ua"]
    assert "text/html" in captured["accept"]


def test_search_duckduckgo_instant_answer_normalizes_topics(monkeypatch) -> None:
    payload = {
        "Heading": "Satya Nadella",
        "AbstractURL": "https://en.wikipedia.org/wiki/Satya_Nadella",
        "AbstractText": "CEO of Microsoft",
        "RelatedTopics": [{"Text": "Satya Nadella - Biography", "FirstURL": "https://example.com/satya"}],
    }

    def _fake_urlopen(_request, timeout: int = 10):
        assert timeout == 10
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(search, "urlopen", _fake_urlopen)

    results = search.search_duckduckgo_instant_answer("Satya Nadella", max_results=5)

    assert len(results) == 2
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Satya_Nadella"
    assert results[1]["title"] == "Satya Nadella"


def test_parse_duckduckgo_html_results_has_fallback_link_extraction() -> None:
    results = search.parse_duckduckgo_html_results(SAMPLE_DDG_HTML_FALLBACK, max_results=10)
    assert results
    assert results[0]["url"] == "https://example.com/in/jane-doe"


def test_parse_duckduckgo_html_results_filters_internal_duckduckgo_pages() -> None:
    results = search.parse_duckduckgo_html_results(SAMPLE_DDG_HTML_INTERNAL, max_results=10)
    assert len(results) == 1
    assert results[0]["url"] == "https://example.net/in/laura-gomez"


def test_enrich_search_results_filters_placeholder_titles_and_preserves_valid_external() -> None:
    enriched = search.enrich_search_results(
        [
            {"title": "here", "url": "https://example.com/profile", "snippet": "placeholder"},
            {"title": "Satya Nadella - Microsoft", "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella", "snippet": "CEO of Microsoft"},
            {"title": "DuckDuckGo privacy", "url": "https://duckduckgo.com/privacy", "snippet": "privacy page"},
        ]
    )
    assert len(enriched) == 1
    assert enriched[0].url == "https://www.microsoft.com/en-us/about/leadership/satya-nadella"


def test_filter_search_results_keeps_high_signal_public_profile() -> None:
    normalized = search.normalize_search_results(
        [
            {
                "title": "Satya Nadella - Chairman and CEO, Microsoft",
                "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
                "snippet": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
            }
        ]
    )
    filtered, decisions = search.filter_search_results(normalized)
    assert len(filtered) == 1
    assert decisions[0].accepted is True
    assert decisions[0].reason == "accepted"


def test_filter_search_results_drops_internal_pages_but_not_external_candidates() -> None:
    normalized = search.normalize_search_results(
        [
            {"title": "Privacy, simplified.", "url": "https://duckduckgo.com/privacy", "snippet": "DuckDuckGo privacy"},
            {"title": "Satya Nadella - Wikipedia", "url": "https://en.wikipedia.org/wiki/Satya_Nadella", "snippet": "Indian-American business executive"},
        ]
    )
    filtered, decisions = search.filter_search_results(normalized)
    assert len(filtered) == 1
    assert filtered[0].url == "https://en.wikipedia.org/wiki/Satya_Nadella"
    assert decisions[0].reason == "search_engine_or_internal"
