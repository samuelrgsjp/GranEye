from graneye.clustering import cluster_identities
from graneye.extraction import extract_candidate_names
from graneye.models import ProfileRecord


def test_multilingual_name_extraction_handles_diacritics_and_cjk() -> None:
    text = "Investigating José Álvarez connections with 山田 太郎 and JOSE ALVAREZ aliases."

    candidates = extract_candidate_names(text, source="unit")
    normalized = {candidate.normalized for candidate in candidates}

    assert "jose alvarez" in normalized
    assert "山田 太郎" in normalized
    assert len([name for name in normalized if name == "jose alvarez"]) == 1


def test_identity_clustering_merges_diacritic_variants() -> None:
    records = [
        ProfileRecord(
            identifier="1",
            display_name="José Álvarez",
            url="https://example.com/profile/jose-alvarez",
            metadata={"handle": "@JoseAlvarez"},
        ),
        ProfileRecord(
            identifier="2",
            display_name="JOSE ALVAREZ",
            url="https://social.example/u/josealvarez",
            metadata={"handle": "jose_alvarez"},
        ),
        ProfileRecord(
            identifier="3",
            display_name="山田 太郎",
            url="https://example.jp/profile/yamada",
            metadata={"handle": "taro_yamada"},
        ),
    ]

    clusters = cluster_identities(records)

    assert len(clusters) == 2
    largest = clusters[0]
    assert largest.key == "jose alvarez"
    assert len(largest.members) == 2
