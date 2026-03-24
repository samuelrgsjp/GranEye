from __future__ import annotations

import graneye
from graneye.analyzers.rule_based import RuleBasedAnalyzer
from graneye.models import ProfileRecord


EXPECTED_EXPORTS = {
    "AnalysisResult",
    "Analyzer",
    "Candidate",
    "IdentityCluster",
    "LocalAnalyzerBackend",
    "LocalAnalyzerInput",
    "ProfileRecord",
    "RuleBasedAnalyzer",
    "analyze_records",
    "cluster_identities",
    "cluster_records",
    "extract_candidate_names",
    "is_directory_url",
}


def test_top_level_exports_remain_available() -> None:
    exported = set(graneye.__all__)
    assert EXPECTED_EXPORTS.issubset(exported)


def test_public_pipeline_flow_stays_usable() -> None:
    records = [
        ProfileRecord(
            identifier="1",
            display_name="Jane Doe",
            url="https://example.com/in/jane-doe",
        ),
        ProfileRecord(
            identifier="2",
            display_name="People Directory",
            url="https://example.com/directory/people",
        ),
    ]

    analyzer = RuleBasedAnalyzer()
    results = graneye.analyze_records(records, analyzer)
    clusters = graneye.cluster_records(records)

    assert len(results) == 1
    assert results[0][0].identifier == "1"
    assert len(clusters) == 2
