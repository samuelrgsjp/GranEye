from __future__ import annotations

from urllib.parse import urlparse

_DIRECTORY_HINTS = {
    "directory",
    "people",
    "profiles",
    "users",
    "members",
    "team",
    "staff",
    "search",
    "listing",
}

_PROFILE_HINTS = {"about", "bio", "u", "user", "profile", "in", "id"}


def is_directory_url(url: str) -> bool:
    """Classify likely directory/listing pages vs single profile pages."""

    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    query = (parsed.query or "").casefold()

    if not segments:
        return False

    lowered_segments = [segment.casefold() for segment in segments]

    if any(hint in lowered_segments for hint in _DIRECTORY_HINTS):
        return True

    if len(segments) == 1 and segments[0].isdigit():
        return False

    if len(segments) <= 2 and any(seg in _PROFILE_HINTS for seg in lowered_segments):
        return False

    # Query-based search pages are usually directory-like.
    if "q=" in query or "search=" in query:
        return True

    # Long trailing slug after an explicit profile token indicates profile.
    if len(segments) >= 2 and lowered_segments[-2] in _PROFILE_HINTS and len(segments[-1]) >= 3:
        return False

    # Heuristic: multiple path segments with no profile token skew to directory pages.
    return len(segments) >= 3
