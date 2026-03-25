from __future__ import annotations

import json
import io
import runpy
import sys
from pathlib import Path

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


def _fake_ranked_with_title(title: str, *, url: str = "https://example.com/profile") -> list[ScoredCandidate]:
    result = SearchResult(
        title=title,
        url=url,
        domain="example.com",
        snippet="Example snippet",
    )
    return [
        ScoredCandidate(
            result=result,
            score=0.7,
            entity_type="unknown",
            name_match="weak_match",
            context_strength=0.0,
            authority_tier="reputable_media",
            seo_penalty=0.0,
            is_noise=False,
            reasons=("entity:unknown",),
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


def test_parse_args_supports_batch_flag_without_alias() -> None:
    parsed = cli._parse_args(["--batch"])
    assert parsed.batch is True


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

    assert code == 2
    assert "Status: no-resolution" in captured.out
    assert "Top candidate:" in captured.out
    assert "Reason: search_only_unverified_candidate" in captured.out


def test_fallback_explanation_is_conservative_and_does_not_claim_fetch_failure() -> None:
    output = cli._build_final_output(
        query_validity="valid",
        resolved=None,
        ranked=_fake_ranked(),
    )
    assert output.explanation == "search-only fallback: ranked evidence available, but identity resolution did not complete."
    assert "fetch failed" not in output.explanation


def test_fallback_normalized_candidate_name_is_empty_for_non_person_article_title() -> None:
    ranked = _fake_ranked_with_title("NVIDIA reports record quarterly revenue in 2026")
    output = cli._build_final_output(
        query_validity="valid",
        resolved=None,
        ranked=ranked,
    )
    assert output.normalized_candidate_name == ""
    assert output.source_title == "NVIDIA reports record quarterly revenue in 2026"
    assert output.no_resolution is True
    assert output.no_resolution_reason == "search_only_unverified_candidate"


def test_fallback_normalized_candidate_name_is_retained_for_person_like_title() -> None:
    ranked = _fake_ranked_with_title("Jensen Huang - Founder and CEO")
    output = cli._build_final_output(
        query_validity="valid",
        resolved=None,
        ranked=ranked,
    )
    assert output.normalized_candidate_name == "jensen huang"
    assert output.source_title == "Jensen Huang - Founder and CEO"
    assert output.no_resolution is True
    assert output.no_resolution_reason == "search_only_unverified_candidate"


def test_main_no_candidates_status_consistent_between_human_and_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (None, []))

    human_code = cli.main(["Laura Gómez Martínez"])
    human = capsys.readouterr()
    assert human_code == 2
    assert "Status: no-resolution" in human.out

    json_code = cli.main(["Laura Gómez Martínez", "--json"])
    machine = capsys.readouterr()
    payload = json.loads(machine.out)
    assert json_code == 2
    assert payload["resolution_status"] == "no-resolution"


def test_carlos_style_case_status_consistent_between_human_and_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = ResolutionOutput(
        normalized_candidate_name="",
        source_url="",
        source_title="",
        final_score=0.49,
        entity_type="institutional_profile",
        same_person_probability=0.49,
        context_match_probability=0.21,
        possible_role="cybersecurity",
        possible_organization=None,
        possible_location="Spain",
        explanation="NO_RESOLUTION: insufficient absolute evidence for unique identity.",
        no_resolution=True,
        no_resolution_reason="insufficient_evidence",
        confidence_label="low",
    )
    ranked = _fake_ranked("https://www.iansresearch.com/our-faculty/faculty/detail/carlos-perez")
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (output, ranked))

    human_code = cli.main(["Carlos Pérez", "Cybersecurity Spain"])
    human = capsys.readouterr()
    assert human_code == 2
    assert "Status: no-resolution" in human.out

    json_code = cli.main(["Carlos Pérez", "Cybersecurity Spain", "--json"])
    machine = capsys.readouterr()
    payload = json.loads(machine.out)
    assert json_code == 2
    assert payload["resolution_status"] == "no-resolution"


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


def test_invalid_query_cannot_be_reported_as_resolved_even_if_resolver_returns_candidate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))

    code = cli.main(["a", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert payload["query_validity"] == "too_short"
    assert payload["resolution_status"] == "no-resolution"
    assert payload["no_resolution_reason"] == "invalid_query:too_short"


def test_repeated_serialization_of_same_final_output_is_stable() -> None:
    finalized = cli.FinalizedQueryResult(
        target_name="Laura Gómez Martínez",
        target_context="Lawyer Barcelona",
        query_validity="valid",
        output=_fake_output(),
        status="resolved",
    )
    first = cli._json_payload(finalized)
    second = cli._json_payload(finalized)
    assert first == second
    assert first["resolution_status"] == "resolved"


def test_repeated_identical_json_runs_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))
    outputs: list[str] = []
    for _ in range(3):
        code = cli.main(["Jensen Huang", "NVIDIA CEO", "--json"])
        captured = capsys.readouterr()
        assert code == 0
        outputs.append(captured.out.strip())
    assert outputs[0] == outputs[1] == outputs[2]


def test_human_and_json_share_status_for_ambiguous_case(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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

    human_code = cli.main(["Jane Doe"])
    human = capsys.readouterr()
    assert human_code == 1
    assert "Status: ambiguous" in human.out

    json_code = cli.main(["Jane Doe", "--json"])
    machine = capsys.readouterr()
    payload = json.loads(machine.out)
    assert json_code == 1
    assert payload["resolution_status"] == "ambiguous"


def test_batch_file_jsonl_processes_multiple_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))
    batch_file = tmp_path / "targets.txt"
    batch_file.write_text(
        "\n# comment\nJensen Huang\tNVIDIA CEO\nCarlos Pérez\tCybersecurity Spain\nSatya Nadella\n",
        encoding="utf-8",
    )

    code = cli.main(["--input-file", str(batch_file), "--jsonl"])
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]

    assert code == 0
    assert len(payloads) == 3
    assert payloads[0]["input_index"] == 1
    assert payloads[2]["target_context"] is None
    assert all(item["error"] is None for item in payloads)


def test_batch_stdin_jsonl_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))
    monkeypatch.setattr(sys, "stdin", io.StringIO("Taylor Swift\tmusic singer usa\nSatya Nadella\n"))

    code = cli.main(["--batch", "--jsonl"])
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert code == 0
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["target_name"] == "Taylor Swift"
    assert second["target_context"] is None


def test_batch_stdin_human_output_has_readable_block_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (_fake_output(), _fake_ranked()))
    monkeypatch.setattr(sys, "stdin", io.StringIO("Jensen Huang\tNVIDIA CEO\nSatya Nadella\n"))

    code = cli.main(["--batch"])
    captured = capsys.readouterr()

    assert code == 0
    assert "[1] Jensen Huang | NVIDIA CEO" in captured.out
    assert "Status: resolved" in captured.out
    assert "Top candidate: laura gomez martinez" in captured.out
    assert "[2] Satya Nadella | (none)" in captured.out
    assert "\n---\n" not in captured.out


def test_batch_human_and_jsonl_share_record_semantics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _resolve(name: str, **_kwargs: object) -> tuple[ResolutionOutput | None, list[ScoredCandidate]]:
        if name == "Carlos Pérez":
            return (
                ResolutionOutput(
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
                    explanation="NO_RESOLUTION: insufficient evidence",
                    no_resolution=True,
                    no_resolution_reason="insufficient_evidence",
                    confidence_label="low",
                ),
                [],
            )
        return _fake_output(), _fake_ranked()

    monkeypatch.setattr(cli, "resolve_query", _resolve)
    monkeypatch.setattr(sys, "stdin", io.StringIO("Jensen Huang\tNVIDIA CEO\nCarlos Pérez\tCybersecurity Spain\n"))
    human_code = cli.main(["--batch"])
    human = capsys.readouterr()
    assert human_code == 0
    assert "[1] Jensen Huang | NVIDIA CEO" in human.out
    assert "Status: resolved" in human.out
    assert "[2] Carlos Pérez | Cybersecurity Spain" in human.out
    assert "Status: no-resolution" in human.out
    assert "Reason: insufficient_evidence" in human.out

    monkeypatch.setattr(sys, "stdin", io.StringIO("Jensen Huang\tNVIDIA CEO\nCarlos Pérez\tCybersecurity Spain\n"))
    json_code = cli.main(["--batch", "--jsonl"])
    machine = capsys.readouterr()
    payloads = [json.loads(line) for line in machine.out.splitlines() if line.strip()]
    assert json_code == 0
    assert payloads[0]["resolution_status"] == "resolved"
    assert payloads[1]["resolution_status"] == "no-resolution"
    assert payloads[1]["no_resolution_reason"] == "insufficient_evidence"


def test_batch_per_record_failure_does_not_abort(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _resolve(name: str, **_kwargs: object) -> tuple[ResolutionOutput, list[ScoredCandidate]]:
        if name == "Bad Name":
            raise RuntimeError("bad record")
        return _fake_output(), _fake_ranked()

    monkeypatch.setattr(cli, "resolve_query", _resolve)
    monkeypatch.setattr(sys, "stdin", io.StringIO("Good Name\tContext\nBad Name\tContext\nAnother Good\n"))

    code = cli.main(["--batch", "--jsonl"])
    captured = capsys.readouterr()
    payloads = [json.loads(line) for line in captured.out.splitlines() if line.strip()]

    assert code == 0
    assert len(payloads) == 3
    assert payloads[0]["error"] is None
    assert payloads[1]["error"] == "bad record"
    assert payloads[2]["error"] is None


def test_batch_requires_batch_output_flags_for_single_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["--batch", "--json"])
    captured = capsys.readouterr()
    assert code == 3
    assert "use --jsonl for batch mode" in captured.err


def test_single_mode_rejects_jsonl_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["Laura Gómez Martínez", "--jsonl"])
    captured = capsys.readouterr()
    assert code == 3
    assert "--jsonl is for batch mode only" in captured.err


def test_main_rejects_json_and_jsonl_together(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["Laura Gómez Martínez", "--json", "--jsonl"])
    captured = capsys.readouterr()
    assert code == 3
    assert "use either --json or --jsonl" in captured.err


def test_batch_file_read_failure_returns_cli_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--input-file", "missing-file.txt", "--jsonl"])
    captured = capsys.readouterr()
    assert code == 3
    assert "failed to read batch input" in captured.err
