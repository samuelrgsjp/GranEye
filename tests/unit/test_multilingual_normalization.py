from __future__ import annotations

import pytest

from graneye.normalization import normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("José Álvarez", "jose alvarez"),
        ("  MARTÍNEZ   GÓMEZ ", "martinez gomez"),
        ("Gómez-Martínez", "gomez martinez"),
        ("山田 太郎", "山田 太郎"),
        ("Renée_O'Connor", "renee o'connor"),
    ],
)
def test_normalize_name_multilingual_and_separators(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        ("José Álvarez", "Jose Alvarez", True),
        ("Gómez Martínez", "Martínez Gómez", False),
        ("Anaïs Nin", "Anais Nin", True),
    ],
)
def test_normalize_name_equivalence_edges(left: str, right: str, equal: bool) -> None:
    left_n = normalize_name(left)
    right_n = normalize_name(right)

    assert (left_n == right_n) is equal
