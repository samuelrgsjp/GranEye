from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import IdentityCluster, ProfileRecord
from .normalization import normalize_handle, normalize_name


def _name_key(value: str) -> str:
    return normalize_name(value)


def _metadata_handle(record: ProfileRecord) -> str:
    handle = record.metadata.get("handle")
    if isinstance(handle, str) and handle.strip():
        return normalize_handle(handle)
    return ""


def _cluster_key(record: ProfileRecord) -> str:
    name = _name_key(record.display_name)
    handle = _metadata_handle(record)

    # If the handle contains the normalized name tokens, prefer the name key.
    if handle and name:
        compact_name = name.replace(" ", "")
        if compact_name and compact_name in handle:
            return name

    return name or handle or record.identifier.casefold()


def cluster_identities(records: Iterable[ProfileRecord]) -> list[IdentityCluster]:
    """Group profile records into identity clusters using deterministic rules."""

    grouped: dict[str, list[ProfileRecord]] = defaultdict(list)
    for record in records:
        key = _cluster_key(record)
        grouped[key].append(record)

    clusters: list[IdentityCluster] = []
    for key, members in grouped.items():
        confidence = min(0.99, 0.65 + 0.1 * (len(members) - 1))
        clusters.append(
            IdentityCluster(
                key=key,
                members=tuple(members),
                confidence=confidence,
            )
        )

    clusters.sort(key=lambda item: (-len(item.members), item.key))
    return clusters
