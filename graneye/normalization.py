from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATORS_RE = re.compile(r"[-_./]+")


def strip_diacritics(value: str) -> str:
    """Drop combining marks while keeping base characters."""

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(value: str) -> str:
    """Normalize free-form names for robust comparison."""

    folded = value.casefold().strip()
    folded = _SEPARATORS_RE.sub(" ", folded)
    folded = _WHITESPACE_RE.sub(" ", folded)
    normalized = strip_diacritics(folded)
    return normalized.strip()


def normalize_handle(value: str) -> str:
    """Normalize social handles and usernames for similarity checks."""

    folded = value.casefold().strip().lstrip("@")
    return _SEPARATORS_RE.sub("", folded)
