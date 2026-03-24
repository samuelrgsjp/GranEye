from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Candidate
from .normalization import normalize_name

_LATIN_NAME_PATTERN = re.compile(
    r"\b(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿā-ž]{1,}|[A-ZÀ-ÖØ-Þ]{2,})(?:[\s'’.-]+(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿā-ž]{1,}|[A-ZÀ-ÖØ-Þ]{2,})){1,2}\b"
)
_CJK_NAME_PATTERN = re.compile(r"[一-龯ぁ-ゔァ-ヴー々〆〤]{1,4}\s+[一-龯ぁ-ゔァ-ヴー々〆〤]{1,4}")

_BLOCKED_TERMS = {
    "directory",
    "index",
    "users",
    "profiles",
    "team",
    "people",
    "admin",
    "support",
    "contact",
}


def _is_likely_name(raw: str) -> bool:
    cleaned = raw.strip(" .-_/")
    if len(cleaned) < 3:
        return False

    normalized = normalize_name(cleaned)
    if not normalized or normalized in _BLOCKED_TERMS:
        return False

    tokens = [token for token in normalized.split(" ") if token]
    if len(tokens) < 2:
        return False

    if all(token.isdigit() for token in tokens):
        return False

    return True


def extract_candidate_names(text: str, *, source: str = "text") -> list[Candidate]:
    """Extract normalized candidate names from free-form text.

    The function is deterministic and dependency-free for easy unit testing.
    """

    candidates: list[Candidate] = []
    seen: set[str] = set()

    matches = list(_LATIN_NAME_PATTERN.finditer(text)) + list(_CJK_NAME_PATTERN.finditer(text))
    for match in matches:
        raw = match.group(0).strip()
        if not _is_likely_name(raw):
            continue

        normalized = normalize_name(raw)
        if normalized in seen:
            continue

        seen.add(normalized)
        token_count = len(normalized.split())
        confidence = min(0.9, 0.6 + token_count * 0.1)

        candidates.append(
            Candidate(
                raw=raw,
                normalized=normalized,
                confidence=confidence,
                source=source,
            )
        )

    return candidates


def extract_from_fields(values: Iterable[str], *, source: str = "record") -> list[Candidate]:
    """Convenience wrapper for extracting candidates across several fields."""

    combined = "\n".join(values)
    return extract_candidate_names(combined, source=source)
