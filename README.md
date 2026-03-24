# GranEye

GranEye is a modular Python OSINT prototype with deterministic analysis and identity resolution.

## Highlights

- Stable analyzer interface (`Analyzer`) with a production-ready rule-based implementation.
- Deterministic candidate name extraction with multilingual support.
- URL heuristics to separate directory listings from profile pages.
- Identity clustering based on normalized names/handles.
- Clean extension contract for future local analyzers without external AI API dependencies.

## Quick Start

```bash
python -m pytest
```

## Package Layout

- `graneye/analyzers/base.py`: analyzer contracts and future local analyzer backend interface.
- `graneye/analyzers/rule_based.py`: existing rule-based analyzer implementation.
- `graneye/extraction.py`: candidate name extraction utilities.
- `graneye/detection.py`: directory vs profile URL classification.
- `graneye/clustering.py`: deterministic identity clustering.
- `graneye/pipeline.py`: composition helpers for analysis and clustering.
