# GranEye

GranEye is a modular Python OSINT prototype with deterministic analysis and identity resolution.

## Highlights

- Stable analyzer interface (`Analyzer`) with a production-ready rule-based implementation.
- Deterministic candidate name extraction with multilingual support.
- URL heuristics to separate directory listings from profile pages.
- Identity clustering based on normalized names/handles.
- Evidence-driven candidate ranking using title/snippet/URL signals.
- Top-candidate page extraction with lightweight rule-based profile inference.
- Graceful fallback paths when target pages block scraping (for example HTTP 999).

## Quick Start

```bash
python -m pytest
```

## CLI Usage

After installing (`pip install -e .`), run either form:

```bash
graneye "Laura Gómez Martínez" "Lawyer Barcelona"
python -m graneye "Laura Gómez Martínez" "Lawyer Barcelona"
```

Notes:
- `target_name` is required.
- `target_context` is optional and can improve disambiguation.
- Use `graneye --help` for argument details.
- CLI output includes `Resolution path` (`full_content`, `partial_content`, `search_only`, or `fetch_blocked`) and `Fetch status`.

## Reliability Behavior

GranEye uses layered fallback so a blocked page fetch does not invalidate ranking:

1. Query DuckDuckGo HTML endpoint, then supplement with Instant Answer topics.
2. Rank all normalized results using deterministic identity/context signals.
3. Attempt to fetch the top candidate page for enrichment.
4. If fetching fails or is blocked, keep ranked search evidence and return a search-grounded result instead of failing the full pipeline.

This means output can still be meaningful even when a website denies bot-like requests.

## Package Layout

- `graneye/analyzers/base.py`: analyzer contracts and future local analyzer backend interface.
- `graneye/analyzers/rule_based.py`: existing rule-based analyzer implementation.
- `graneye/clustering.py`: deterministic identity clustering.
- `graneye/cli.py`: command-line entrypoint and output rendering.
- `graneye/detection.py`: directory vs profile URL classification.
- `graneye/extraction.py`: candidate name extraction utilities.
- `graneye/pipeline.py`: composition helpers for analysis, clustering, and query resolution orchestration.
- `graneye/search.py`: normalization/enrichment for raw search results.
- `graneye/resolution.py`: candidate ranking, scoring rationale, top-page extraction, and structured output.
