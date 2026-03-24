from graneye.detection import is_directory_url


def test_detects_directory_listing_paths() -> None:
    assert is_directory_url("https://target.example/directory/people")
    assert is_directory_url("https://target.example/users?search=jane")
    assert is_directory_url("https://target.example/company/people/engineering")


def test_detects_single_profile_paths() -> None:
    assert not is_directory_url("https://target.example/profile/jane-doe")
    assert not is_directory_url("https://target.example/u/janedoe")
    assert not is_directory_url("https://target.example/in/jane-doe")
