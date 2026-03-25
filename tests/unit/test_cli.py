from __future__ import annotations

import json
import runpy

import pytest

from graneye import cli
from graneye.resolution import ResolutionOutput, ScoredCandidate
from graneye.search import SearchResult


def _fake_ranked(url: str = "https://example.com/in/laura") -> list[ScoredCandidate]:
    result = SearchResult(
        title="Laura Gómez Martínez - Lawyer",
        url=url,
        domain="example.com",
        snippet="Lawyer in Barcelona",
    )
    return [
        ScoredCandidate(
            result=result,
            score=0.92,
            entity_type="person_profile",
            name_match="full_match",
            context_strength=0.45,
            authority_tier="public_structured_profile",
            seo_penalty=0.0,
            is_noise=False,
            reasons=("entity:person_profile", "name:full_match"),
        )
    ]


def _fake_output(url: str = "https://example.com/in/laura") -> ResolutionOutput:
    return ResolutionOutput(
        normalized_candidate_name="laura gomez martinez",
        source_url=url,
        source_title="Laura Gómez Martínez - Lawyer",
        final_score=0.92,
        entity_type="person_profile",
        same_person_probability=0.92,
        context_match_probability=0.45,
        possible_role="lawyer",
        possible_organization="Acme Legal",
        possible_location="Barcelona",
        explanation="Selected example.com with full_match and person_profile",
    )


def test_parse_args_supports_optional_context() -> None:
    parsed = cli._parse_args(["Laura Gómez Martínez", "Lawyer Barcelona"])
    assert parsed.target_name == "Laura Gómez Martínez"
    assert parsed.target_context == "Lawyer Barcelona"
    assert parsed.debug is False
    assert parsed.json is False


def test_parse_args_supports_debug_flag() -> None:
    parsed = cli._parse_args(["Laura Gómez Martínez", "Lawyer Barcelona", "--debug"])
    assert parsed.debug is True


def test_parse_args_supports_json_flag() -> None:
    parsed = cli._parse_args(["Laura Gómez Martínez", "--json"])
    assert parsed.json is True


@pytest.mark.parametrize(
    ("argv", "expected_code", "expected_text"),
    [
        (["Laura Gómez Martínez", "Lawyer Barcelona"], 0, "Top candidate: laura gomez martinez"),
        (["Laura Gómez Martínez"], 0, "Context: (none)"),
    ],
)
def test_main_prints_candidate_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected_code: int,
    expected_text: str,
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))

    code = cli.main(argv)
    captured = capsys.readouterr()

    assert code == expected_code
    assert expected_text in captured.out
    assert "Status: resolved" in captured.out
    assert "Source URL:" in captured.out
    assert "Score:" not in captured.out
    assert "Resolution path:" not in captured.out
    assert "Decision reason:" not in captured.out


def test_main_returns_not_found_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (None, []))

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Status: no-resolution" in captured.out


def test_main_falls_back_to_search_only_when_resolution_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (None, _fake_ranked()))

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Status: resolved" in captured.out
    assert "Top candidate:" in captured.out


def test_main_handles_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[None, list[ScoredCandidate]]:
        raise RuntimeError("network down")

    monkeypatch.setattr(cli, "resolve_query", _boom)

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 3
    assert "failed to execute pipeline" in captured.err


def test_module_execution_calls_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}

    def _fake_main() -> int:
        called["value"] = True
        return 0

    monkeypatch.setattr("graneye.cli.main", _fake_main)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("graneye", run_name="__main__")

    assert called["value"] is True
    assert exc.value.code == 0


def test_main_rejects_blank_target_name(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["   "])
    captured = capsys.readouterr()

    assert code == 3
    assert "target_name must not be empty" in captured.err


def test_main_debug_prints_pipeline_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostics = cli.SearchPipelineDiagnostics(
        raw_results_count=5,
        normalized_results_count=5,
        filtered_results_count=3,
        ranked_candidates_count=3,
        filter_decisions=(),
        ranked_candidates=tuple(_fake_ranked()),
        query_attempts=("Laura Gómez Martínez Lawyer Barcelona",),
    )

    monkeypatch.setattr(cli, "resolve_query_with_debug", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked(), diagnostics))

    code = cli.main(["Laura Gómez Martínez", "Lawyer Barcelona", "--debug"])
    captured = capsys.readouterr()

    assert code == 0
    assert "=== Debug: Search pipeline ===" in captured.out
    assert "Query attempts:" in captured.out
    assert "Raw results count: 5" in captured.out
    assert "Filtered results count: 3" in captured.out
    assert "Ranked candidates:" in captured.out
    assert "Decision reason:" in captured.out


def test_main_uses_selected_representative_title_not_ranked_first_title(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _fake_output("https://www.nvidia.com/en-us/about-nvidia/jensen-huang/")
    output = ResolutionOutput(
        normalized_candidate_name=output.normalized_candidate_name,
        source_url=output.source_url,
        source_title="Jensen Huang - Founder and CEO",
        final_score=output.final_score,
        entity_type=output.entity_type,
        same_person_probability=output.same_person_probability,
        context_match_probability=output.context_match_probability,
        possible_role=output.possible_role,
        possible_organization=output.possible_organization,
        possible_location=output.possible_location,
        explanation=output.explanation,
    )
    ranked = _fake_ranked("https://business-news-today.example.com/about-nvidia")
    diagnostics = cli.SearchPipelineDiagnostics(
        raw_results_count=1,
        normalized_results_count=1,
        filtered_results_count=1,
        ranked_candidates_count=1,
        filter_decisions=(),
        ranked_candidates=tuple(ranked),
    )
    monkeypatch.setattr(cli, "resolve_query_with_debug", lambda *_args, **_kwargs: (output, ranked, diagnostics))

    code = cli.main(["Jensen Huang", "NVIDIA CEO", "--debug"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Display title: Jensen Huang - Founder and CEO" in captured.out


def test_main_returns_ambiguous_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    output = ResolutionOutput(
        normalized_candidate_name="jane doe",
        source_url="https://example.org/jane",
        source_title="Jane Doe - Example",
        final_score=0.62,
        entity_type="person_profile",
        same_person_probability=0.62,
        context_match_probability=0.30,
        possible_role=None,
        possible_organization=None,
        possible_location=None,
        explanation="AMBIGUOUS: close candidates.",
        ambiguity_detected=True,
        ambiguity_reason="multiple_plausible_candidates",
        no_resolution=False,
        no_resolution_reason=None,
    )
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (output, _fake_ranked()))

    code = cli.main(["Jane Doe"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Status: ambiguous" in captured.out


def test_main_json_output_has_expected_keys(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))

    code = cli.main(["Laura Gómez Martínez", "Lawyer Barcelona", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    expected_keys = {
        "target_name",
        "target_context",
        "query_validity",
        "resolution_status",
        "no_resolution_reason",
        "ambiguity_reason",
        "confidence",
        "top_candidate",
        "source_url",
        "display_title",
        "same_person_probability",
        "context_match_probability",
        "entity_type",
        "decision_reason",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["resolution_status"] == "resolved"


@pytest.mark.parametrize(
    ("output", "expected_status"),
    [
        (_fake_output(), "resolved"),
        (
            ResolutionOutput(
                normalized_candidate_name="",
                source_url="",
                source_title="",
                final_score=0.41,
                entity_type="unknown",
                same_person_probability=0.0,
                context_match_probability=0.1,
                possible_role=None,
                possible_organization=None,
                possible_location=None,
                explanation="NO_RESOLUTION: insufficient evidence",
                no_resolution=True,
                no_resolution_reason="insufficient_evidence",
                confidence_label="low",
            ),
            "no-resolution",
        ),
    ],
)
def test_human_and_json_share_resolution_status_mapping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: ResolutionOutput,
    expected_status: str,
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (output, _fake_ranked()))

    code = cli.main(["Laura Gómez Martínez"])
    human = capsys.readouterr()
    assert code == (0 if expected_status == "resolved" else 2)
    assert f"Status: {expected_status}" in human.out

    code = cli.main(["Laura Gómez Martínez", "--json"])
    machine = capsys.readouterr()
    payload = json.loads(machine.out)
    assert code == (0 if expected_status == "resolved" else 2)
    assert payload["resolution_status"] == expected_status


@pytest.mark.parametrize("query", ["123456", "a"])
def test_invalid_query_reports_no_resolution_in_human_and_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    query: str,
) -> None:
    invalid_output = ResolutionOutput(
        normalized_candidate_name="",
        source_url="",
        source_title="",
        final_score=0.0,
        entity_type="unknown",
        same_person_probability=0.0,
        context_match_probability=0.0,
        possible_role=None,
        possible_organization=None,
        possible_location=None,
        explanation=f"NO_RESOLUTION: invalid query ({query})",
        no_resolution=True,
        no_resolution_reason="invalid_query:numeric_or_garbage",
        ambiguity_detected=False,
        ambiguity_reason=None,
        confidence_label="low",
    )
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (invalid_output, _fake_ranked()))

    human_code = cli.main([query])
    human = capsys.readouterr()
    assert human_code == 2
    assert "Status: no-resolution" in human.out

    json_code = cli.main([query, "--json"])
    machine = capsys.readouterr()
    payload = json.loads(machine.out)
    assert json_code == 2
    assert payload["resolution_status"] == "no-resolution"
