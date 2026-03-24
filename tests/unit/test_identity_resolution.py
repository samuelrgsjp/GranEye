from __future__ import annotations

import pytest

from graneye.clustering import cluster_identities
from graneye.models import ProfileRecord
from graneye.normalization import normalize_name


@pytest.mark.parametrize(
    ("left", "right", "should_merge"),
    [
        ("Gómez Martínez", "Martínez Gómez", True),
        ("José Álvarez", "JOSE ALVAREZ", True),
        ("Jane Mary Doe", "Doe Jane Mary", True),
        ("John Smith", "Johnny Smith", False),
        ("Ana Pérez", "Ana Pereira", False),
    ],
)
def test_identity_resolution_name_order_and_false_positives(
    left: str,
    right: str,
    should_merge: bool,
) -> None:
    records = [
        ProfileRecord(identifier="1", display_name=left, url="https://example.com/a"),
        ProfileRecord(identifier="2", display_name=right, url="https://example.com/b"),
    ]

    clusters = cluster_identities(records)

    if should_merge:
        assert len(clusters) == 1
        assert len(clusters[0].members) == 2
    else:
        assert len(clusters) == 2


@pytest.mark.parametrize(
    ("display_name", "handle", "expected_cluster_key"),
    [
        ("José Álvarez", "@josealvarez", "alvarez jose"),
        ("Martínez Gómez", "martinez_gomez", "gomez martinez"),
    ],
)
def test_identity_resolution_handle_and_name_alignment(
    display_name: str,
    handle: str,
    expected_cluster_key: str,
) -> None:
    records = [
        ProfileRecord(
            identifier="1",
            display_name=display_name,
            url="https://example.com/a",
            metadata={"handle": handle},
        ),
        ProfileRecord(
            identifier="2",
            display_name=normalize_name(display_name),
            url="https://example.com/b",
            metadata={"handle": handle},
        ),
    ]

    clusters = cluster_identities(records)

    assert len(clusters) == 1
    assert clusters[0].key == expected_cluster_key


def test_partial_match_not_merged_without_additional_signal() -> None:
    records = [
        ProfileRecord(identifier="1", display_name="María Gómez", url="https://example.com/a"),
        ProfileRecord(identifier="2", display_name="María", url="https://example.com/b"),
    ]

    clusters = cluster_identities(records)

    assert len(clusters) == 2
