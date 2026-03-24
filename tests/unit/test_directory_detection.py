from __future__ import annotations

import pytest

from graneye.detection import is_directory_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://target.example/directory/people", True),
        ("https://target.example/users?search=jane", True),
        ("https://target.example/company/people/engineering", True),
        ("https://www.linkedin.com/search/results/people/?keywords=jane", True),
        ("https://www.linkedin.com/company/acme/people/", True),
        ("https://target.example/profile/jane-doe", False),
        ("https://target.example/u/janedoe", False),
        ("https://www.linkedin.com/in/jane-doe-123456/", False),
        ("https://target.example/in/jane-doe", False),
    ],
)
def test_directory_detection_parametrized(url: str, expected: bool) -> None:
    assert is_directory_url(url) is expected
