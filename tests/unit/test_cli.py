from __future__ import annotations

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
            is_noise=False,
            reasons=("entity:person_profile", "name:full_match"),
        )
    ]


def _fake_output(url: str = "https://example.com/in/laura") -> ResolutionOutput:
    return ResolutionOutput(
        normalized_candidate_name="laura gomez martinez",
        source_url=url,
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


def test_parse_args_supports_debug_flag() -> None:
    parsed = cli._parse_args(["Laura Gómez Martínez", "Lawyer Barcelona", "--debug"])
    assert parsed.debug is True


@pytest.mark.parametrize(
    ("argv", "expected_code", "expected_text"),
    [
        (["Laura Gómez Martínez", "Lawyer Barcelona"], 0, "Top candidate: laura gomez martinez"),
        (["Laura Gómez Martínez"], 0, "Target context: (none)"),
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
    assert "Source URL:" in captured.out
    assert "Score:" in captured.out
    assert "Resolution path:" in captured.out
    assert "Decision reason:" in captured.out


def test_main_returns_not_found_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (None, []))

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 3
    assert "No candidates found" in captured.out


def test_main_falls_back_to_search_only_when_resolution_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "resolve_query", lambda *_args, **_kwargs: (None, _fake_ranked()))

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Top candidate (search-only):" in captured.out
    assert "using ranked search evidence only" in captured.out


def test_main_handles_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> tuple[None, list[ScoredCandidate]]:
        raise RuntimeError("network down")

    monkeypatch.setattr(cli, "resolve_query", _boom)

    code = cli.main(["Laura Gómez Martínez"])
    captured = capsys.readouterr()

    assert code == 1
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

    assert code == 2
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
