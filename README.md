# GranEye

GranEye is a modular Python OSINT prototype with deterministic analysis and identity resolution.

## Highlights

- Stable analyzer interface (`Analyzer`) with a production-ready rule-based implementation.
- Deterministic candidate name extraction with multilingual support.
- URL heuristics to separate directory listings from profile pages.
- Identity clustering based on normalized names/handles.
- Evidence-driven candidate ranking using title/snippet/URL signals.
- Top-candidate page extraction with lightweight rule-based profile inference.

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
