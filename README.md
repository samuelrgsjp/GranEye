# GranEye

GranEye is a modular Python OSINT prototype for lawful, explainable public-identity resolution from open-web evidence.

## Highlights

- Stable analyzer interface (`Analyzer`) with a production-ready rule-based implementation.
- Deterministic candidate name extraction with multilingual support.
- URL heuristics to separate directory listings from profile pages.
- Identity clustering based on normalized names/handles.
- Evidence-driven candidate ranking across professional, academic, institutional, media, creator, and mixed public contexts.
- Top-candidate page extraction with lightweight rule-based profile inference.
- Graceful fallback paths when target pages block scraping (for example HTTP 999).

## Quick Start

```bash
pip install -e .
python -m pytest
```

## CLI Usage

After installing (`pip install -e .`), run either form:

```bash
graneye "Laura Gómez Martínez" "Lawyer Barcelona"
python -m graneye "Laura Gómez Martínez" "Lawyer Barcelona"
```

Notes:
- `target_name` is required in single-query mode.
- `target_context` is optional and can be broad (for example: `Lawyer Barcelona`, `Actress`, `YouTuber Spain`, `University professor Madrid`, `TV personality`).
- Use `graneye --help` for argument details.
- Default CLI output is concise and human-readable (`Name`, `Context`, `Status`, `Confidence`, `Top candidate`, `Source URL`; `Reason` appears for `ambiguous`/`no-resolution`).
- `--debug` keeps the existing detailed diagnostics and verbose decision fields (including resolution path, fetch status, and scoring details).
- `--json` returns stable machine-readable output with keys:
  - `target_name`, `target_context`, `query_validity`, `resolution_status`
  - `no_resolution_reason`, `ambiguity_reason`, `confidence`
  - `top_candidate`, `source_url`, `display_title`
  - `same_person_probability`, `context_match_probability`, `entity_type`, `decision_reason`

### Batch Usage

GranEye now supports explicit batch processing without changing single-query behavior.

#### Input modes

1) File input:

```bash
python -m graneye --input-file targets.txt --jsonl
python -m graneye --input-file targets.csv --jsonl
```

2) Stdin input:

```bash
cat targets.txt | python -m graneye --batch --jsonl
printf 'Jensen Huang\tNVIDIA CEO\nSatya Nadella\n' | python -m graneye --batch --jsonl
```

`--batch` is the single stdin batch-mode switch.

#### Batch input format rules (plain text)

- Each record is one line.
- Format: `target_name[TAB]target_context`
- `target_context` is optional.
- Blank lines are ignored.
- Lines starting with `#` are ignored.

Examples:

```text
Jensen Huang	NVIDIA CEO
Carlos Pérez	Cybersecurity Spain
Taylor Swift	music singer usa
Satya Nadella
```

#### Batch output

- Human-readable batch output (default in batch mode): compact per-record blocks.
- `--jsonl` output (recommended for pipelines): one JSON object per input record.
- `--json` remains single-query only.

Human-readable example:

```text
[1] Jensen Huang | NVIDIA CEO
Status: resolved
Confidence: high
Top candidate: jensen huang
Source URL: https://www.nvidia.com/en-us/about-nvidia/jensen-huang/

[2] Carlos Pérez | Cybersecurity Spain
Status: no-resolution
Confidence: low
Top candidate: (none)
Source URL: (none)
Reason: insufficient_evidence
```

Batch JSONL objects include:

- `input_index`
- `target_name`, `target_context`, `query_validity`
- `resolution_status`, `no_resolution_reason`, `ambiguity_reason`, `confidence`
- `top_candidate`, `source_url`, `display_title`
- `same_person_probability`, `context_match_probability`, `entity_type`, `decision_reason`
- `error` (null unless processing that record failed)

### Resolution States and Exit Codes

- `resolved` → process exit code `0`
- `ambiguous` → process exit code `1`
- `no-resolution` → process exit code `2`
- invalid usage / CLI runtime error → process exit code `3`

Batch mode exit codes:

- batch execution completed (including per-record no-resolution/ambiguity/failures) → process exit code `0`
- batch-level input/CLI/runtime failure that prevents normal processing → process exit code `3`

### Source Categories

GranEye classifies public-web candidates into explainable source types:

- `person_profile`
- `official_profile`
- `official_bio`
- `academic_profile`
- `institutional_profile`
- `media_profile`
- `creator_profile`
- `article`
- `directory`
- `aggregator`
- `unknown`

This enables context-sensitive ranking rather than fixed professional-only bias.

## Reliability Behavior

GranEye uses layered fallback so a blocked page fetch does not invalidate ranking:

1. Run multiple query variants (`name + context`, quoted name + context, and name-only), using DuckDuckGo HTML with Lite fallback plus Instant Answer topics.
2. Rank all normalized results using deterministic identity/context signals with source-type balancing.
3. Attempt to fetch the top candidate page for enrichment.
4. If fetching fails or is blocked, keep ranked search evidence and return a search-grounded result instead of failing the full pipeline.

This means output can still be meaningful even when a website denies bot-like requests.

## Confidence and Ambiguity

- GranEye reports `high`, `medium`, or `low` confidence.
- Ambiguity remains explicit: weak evidence and close candidate competition are surfaced instead of forced certainty.
- Low-confidence outputs are expected for broad/banal names or no-context queries.

## Legal and Data Boundaries

- GranEye is designed for public-web evidence only.
- It does not use external AI APIs, private data brokers, hidden data extraction, or non-public account access.
- Use it only for lawful and legitimate identity-resolution workflows.

## Current Limitations

- Search quality still depends on third-party search result markup and may vary by geography or anti-bot behavior.
- Context parsing is rule-based (not semantic NLP), so very long or unusual context prompts may underperform.
- Page enrichment intentionally uses lightweight parsing; heavily client-rendered pages may return partial signals.

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
