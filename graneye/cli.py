from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from .pipeline import SearchPipelineDiagnostics, resolve_query, resolve_query_with_debug
from .resolution import ResolutionOutput, ScoredCandidate, assess_query_validity
from .search import search_duckduckgo_html, search_duckduckgo_instant_answer


class CLIArgs(argparse.Namespace):
    target_name: str | None
    target_context: str | None
    debug: bool
    json: bool
    jsonl: bool
    input_file: str | None
    batch: bool


@dataclass(frozen=True)
class BatchRecord:
    input_index: int
    target_name: str
    target_context: str | None


@dataclass(frozen=True)
class FinalizedQueryResult:
    target_name: str
    target_context: str | None
    query_validity: str
    output: ResolutionOutput
    status: str


def _parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="graneye",
        description="Resolve a likely public profile candidate for a person name.",
    )
    parser.add_argument("target_name", nargs="?", default=None, help="Target person name to resolve")
    parser.add_argument(
        "target_context",
        nargs="?",
        default=None,
        help="Optional disambiguation context (for example: 'Lawyer Barcelona')",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show search pipeline diagnostics and per-result filter decisions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return stable machine-readable JSON output.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Return newline-delimited JSON output for batch mode.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Read batch input from a file (plain text tab-separated or CSV).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode and read records from stdin.",
    )

    namespace = parser.parse_args(argv, namespace=CLIArgs())
    return namespace


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _render_output(result: FinalizedQueryResult) -> str:
    status = result.status
    if status != "resolved":
        lines = [
            f"Name: {result.target_name}",
            f"Context: {result.target_context}" if result.target_context else "Context: (none)",
            f"Status: {status}",
            f"Confidence: {result.output.confidence_label}",
            f"Top candidate: {result.output.normalized_candidate_name or '(none)'}",
            f"Source URL: {result.output.source_url or '(none)'}",
            f"Reason: {result.output.no_resolution_reason or result.output.ambiguity_reason or result.output.explanation}",
        ]
        return "\n".join(lines)

    lines = [
        f"Name: {result.target_name}",
        f"Context: {result.target_context}" if result.target_context else "Context: (none)",
        "Status: resolved",
        f"Confidence: {result.output.confidence_label}",
        f"Top candidate: {result.output.normalized_candidate_name or '(unknown)'}",
        f"Source URL: {result.output.source_url}",
    ]
    return "\n".join(lines)


def _render_batch_human_output(
    *,
    input_index: int,
    result: FinalizedQueryResult,
    error: str | None,
) -> str:
    lines = [
        f"[{input_index}] {result.target_name} | {result.target_context or '(none)'}",
        f"Status: {result.status}",
        f"Confidence: {result.output.confidence_label}",
        f"Top candidate: {result.output.normalized_candidate_name or '(none)'}",
        f"Source URL: {result.output.source_url or '(none)'}",
    ]
    reason = result.output.no_resolution_reason or result.output.ambiguity_reason
    if reason:
        lines.append(f"Reason: {reason}")
    if error:
        lines.append(f"Error: {error}")
    return "\n".join(lines)


def _render_debug_result_output(result: FinalizedQueryResult) -> str:
    status = result.status
    if status != "resolved":
        lines = [
            f"Target name: {result.target_name}",
            f"Target context: {result.target_context}" if result.target_context else "Target context: (none)",
            f"Resolution: {status}",
            f"Reason: {result.output.no_resolution_reason or result.output.ambiguity_reason or 'unknown'}",
            f"Confidence: {result.output.confidence_label}",
            f"Decision reason: {result.output.explanation}",
        ]
        return "\n".join(lines)

    lines = [
        f"Target name: {result.target_name}",
        f"Target context: {result.target_context}" if result.target_context else "Target context: (none)",
        f"Top candidate: {result.output.normalized_candidate_name or '(unknown)'}",
        f"Source URL: {result.output.source_url}",
        f"Display title: {result.output.source_title or '(not available)'}",
        f"Score: {result.output.final_score:.3f}",
        f"Same-person probability: {result.output.same_person_probability:.3f}",
        f"Context match probability: {result.output.context_match_probability:.3f}",
        f"Confidence: {result.output.confidence_label}",
        f"Ambiguity detected: {'yes' if result.output.ambiguity_detected else 'no'}",
        f"Ambiguity reason: {result.output.ambiguity_reason or '(none)'}",
        f"Resolution path: {result.output.resolution_path}",
        f"Fetch status: {result.output.fetch_status}",
        f"Decision reason: {result.output.explanation}",
    ]
    return "\n".join(lines)


def _resolution_status(output: ResolutionOutput | None) -> str:
    if output is None:
        return "no-resolution"
    if output.no_resolution:
        return "no-resolution"
    if output.ambiguity_detected:
        return "ambiguous"
    return "resolved"


def _build_final_output(
    *,
    query_validity: str,
    resolved: ResolutionOutput | None,
    ranked: list[ScoredCandidate],
) -> ResolutionOutput:
    if resolved is not None:
        return resolved
    if query_validity != "valid":
        return ResolutionOutput(
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
            explanation=f"NO_RESOLUTION: invalid query ({query_validity}); hard rejection triggered.",
            resolution_path="search_only",
            fetch_status="not_attempted",
            confidence_label="low",
            ambiguity_detected=False,
            ambiguity_reason=None,
            no_resolution=True,
            no_resolution_reason=f"invalid_query:{query_validity}",
        )
    if not ranked:
        return ResolutionOutput(
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
            explanation="NO_RESOLUTION: no candidates found",
            resolution_path="search_only",
            fetch_status="not_attempted",
            confidence_label="low",
            ambiguity_detected=False,
            ambiguity_reason=None,
            no_resolution=True,
            no_resolution_reason="no_candidates",
        )

    # Intentional fallback path:
    # If ranking produced candidates but page-level resolution did not complete
    # (for example due to fetch blocking), return a search-only low-confidence
    # output so callers can still inspect the strongest ranked candidate.
    # This is not page-validated identity resolution.
    top = ranked[0]
    return ResolutionOutput(
        normalized_candidate_name=top.result.title or top.result.domain,
        source_url=top.result.url,
        source_title=top.result.title or "",
        final_score=top.score,
        entity_type=top.entity_type,
        same_person_probability=top.score,
        context_match_probability=top.context_strength,
        possible_role=None,
        possible_organization=None,
        possible_location=None,
        explanation="content fetch failed; using ranked search evidence only.",
        resolution_path="search_only",
        fetch_status="not_attempted",
        confidence_label="low",
        ambiguity_detected=False,
        ambiguity_reason=None,
        no_resolution=False,
        no_resolution_reason=None,
    )


def _json_payload(result: FinalizedQueryResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_name": result.target_name,
        "target_context": result.target_context,
        "query_validity": result.query_validity,
        "resolution_status": result.status,
        "no_resolution_reason": result.output.no_resolution_reason,
        "ambiguity_reason": result.output.ambiguity_reason,
        "confidence": result.output.confidence_label,
        "top_candidate": result.output.normalized_candidate_name or None,
        "source_url": result.output.source_url or None,
        "display_title": result.output.source_title or None,
        "same_person_probability": result.output.same_person_probability,
        "context_match_probability": result.output.context_match_probability,
        "entity_type": result.output.entity_type,
        "decision_reason": result.output.explanation,
    }
    return payload


def _batch_json_payload(
    *,
    input_index: int,
    result: FinalizedQueryResult,
    error: str | None = None,
) -> dict[str, object]:
    payload = _json_payload(result)
    payload["input_index"] = input_index
    payload["error"] = error
    return payload


def _iterate_batch_text_lines(lines: Iterable[str]) -> Iterable[tuple[str, str | None]]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name_part, context_part = line.split("\t", 1)
            name = name_part.strip()
            context = context_part.strip() or None
        else:
            name = line
            context = None
        if not name:
            continue
        yield (name, context)


def _iterate_batch_csv_rows(lines: Iterable[str]) -> Iterable[tuple[str, str | None]]:
    reader = csv.reader(lines)
    for row in reader:
        if not row:
            continue
        first = row[0].strip() if row[0] else ""
        if not first or first.startswith("#"):
            continue
        context = row[1].strip() if len(row) > 1 and row[1] else None
        yield (first, context or None)


def _read_batch_records(args: CLIArgs) -> list[BatchRecord]:
    source_lines: list[str]
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as handle:
            source_lines = handle.readlines()
        is_csv = args.input_file.lower().endswith(".csv")
    else:
        source_lines = sys.stdin.read().splitlines()
        is_csv = False

    rows = _iterate_batch_csv_rows(source_lines) if is_csv else _iterate_batch_text_lines(source_lines)
    return [
        BatchRecord(input_index=index, target_name=name, target_context=context)
        for index, (name, context) in enumerate(rows, start=1)
    ]


def _process_query(target_name: str, context: str | None, debug: bool) -> tuple[FinalizedQueryResult, SearchPipelineDiagnostics | None]:
    diagnostics: SearchPipelineDiagnostics | None = None
    if debug:
        resolved_output, ranked, diagnostics = resolve_query_with_debug(
            target_name,
            context=context,
            html_search=search_duckduckgo_html,
            instant_search=search_duckduckgo_instant_answer,
        )
    else:
        resolved_output, ranked = resolve_query(
            target_name,
            context=context,
            html_search=search_duckduckgo_html,
            instant_search=search_duckduckgo_instant_answer,
        )
    query_validity = assess_query_validity(target_name).status
    final_output = _build_final_output(
        query_validity=query_validity,
        resolved=resolved_output,
        ranked=ranked,
    )
    return FinalizedQueryResult(
        target_name=target_name,
        target_context=context,
        query_validity=query_validity,
        output=final_output,
        status=_resolution_status(final_output),
    ), diagnostics


def _empty_error_output(error_text: str) -> ResolutionOutput:
    return ResolutionOutput(
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
        explanation=f"NO_RESOLUTION: {error_text}",
        resolution_path="search_only",
        fetch_status="not_attempted",
        confidence_label="low",
        ambiguity_detected=False,
        ambiguity_reason=None,
        no_resolution=True,
        no_resolution_reason="execution_error",
    )


def _exit_code(status: str) -> int:
    if status == "ambiguous":
        return 1
    if status == "no-resolution":
        return 2
    return 0


def _validate_mode_flags(args: CLIArgs, *, batch_mode: bool) -> int | None:
    if args.json and args.jsonl:
        print("Error: use either --json or --jsonl, not both.", file=sys.stderr)
        return 3
    if batch_mode and args.json:
        print("Error: --json is for single-query mode; use --jsonl for batch mode.", file=sys.stderr)
        return 3
    if not batch_mode and args.jsonl:
        print("Error: --jsonl is for batch mode only.", file=sys.stderr)
        return 3
    return None


def _run_batch_mode(args: CLIArgs) -> int:
    if args.target_name is not None:
        print("Error: positional target_name is not used in batch mode.", file=sys.stderr)
        return 3
    if args.debug:
        print("Error: --debug is only available in single-query mode.", file=sys.stderr)
        return 3
    try:
        records = _read_batch_records(args)
    except Exception as exc:
        print(f"Error: failed to read batch input: {exc}", file=sys.stderr)
        return 3

    for record in records:
        finalized: FinalizedQueryResult
        try:
            finalized, _ = _process_query(record.target_name, record.target_context, debug=False)
            payload = _batch_json_payload(
                input_index=record.input_index,
                result=finalized,
                error=None,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            finalized = FinalizedQueryResult(
                target_name=record.target_name,
                target_context=record.target_context,
                query_validity="unknown",
                output=_empty_error_output(str(exc)),
                status="no-resolution",
            )
            payload = _batch_json_payload(
                input_index=record.input_index,
                result=finalized,
                error=str(exc),
            )
        if args.jsonl:
            print(json.dumps(payload, ensure_ascii=False))
            continue

        print(
            _render_batch_human_output(
                input_index=record.input_index,
                result=finalized,
                error=payload["error"] if isinstance(payload["error"], str) else None,
            )
        )
        print()
    return 0


def _run_single_mode(args: CLIArgs) -> int:
    if _is_blank(args.target_name):
        print("Error: target_name must not be empty.", file=sys.stderr)
        return 3

    target_name = args.target_name.strip()
    context = args.target_context.strip() if args.target_context else None

    diagnostics: SearchPipelineDiagnostics | None = None
    try:
        finalized, diagnostics = _process_query(target_name, context, args.debug)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error: failed to execute pipeline: {exc}", file=sys.stderr)
        return 3

    if args.debug and diagnostics is not None:
        print(_render_debug_output(diagnostics))

    if args.json:
        payload = _json_payload(finalized)
        print(json.dumps(payload, ensure_ascii=False))
        return _exit_code(finalized.status)

    print(
        _render_debug_result_output(finalized)
        if args.debug
        else _render_output(finalized)
    )
    return _exit_code(finalized.status)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    batch_mode = bool(args.input_file or args.batch)
    validation_error = _validate_mode_flags(args, batch_mode=batch_mode)
    if validation_error is not None:
        return validation_error

    return _run_batch_mode(args) if batch_mode else _run_single_mode(args)


def _render_debug_output(diagnostics: SearchPipelineDiagnostics) -> str:
    lines = [
        "=== Debug: Search pipeline ===",
        f"Query attempts: {' | '.join(diagnostics.query_attempts) if diagnostics.query_attempts else '(none)'}",
        f"Raw results count: {diagnostics.raw_results_count}",
        f"Normalized results count: {diagnostics.normalized_results_count}",
        f"Filtered results count: {diagnostics.filtered_results_count}",
        f"Ranked candidates count: {diagnostics.ranked_candidates_count}",
        f"Source diversity (domains): {diagnostics.source_diversity_count}",
        f"Query validity: {diagnostics.query_validity}",
        f"Context interpretation: {diagnostics.context_interpretation or '(none)'}",
        f"Ambiguity triggered: {'yes' if diagnostics.ambiguity_triggered else 'no'}",
        f"Ambiguity reason: {diagnostics.ambiguity_reason or '(none)'}",
        "Discarded candidates:",
    ]
    discarded = [decision for decision in diagnostics.filter_decisions if not decision.accepted]
    kept = [decision for decision in diagnostics.filter_decisions if decision.accepted]
    for idx, decision in enumerate(discarded, start=1):
        result = decision.result
        snippet_flag = "yes" if result.snippet and result.snippet.strip() else "no"
        lines.append(
            (
                f"{idx:02d}. [dropped] title={result.title or '(empty)'} | "
                f"url={result.url or '(empty)'} | domain={result.domain or '(empty)'} | "
                f"snippet={snippet_flag} | reason={decision.reason}"
            )
        )
    if not discarded:
        lines.append("00. [dropped] (none)")
    lines.append("Retained pre-ranking candidates:")
    for idx, decision in enumerate(kept, start=1):
        result = decision.result
        snippet_flag = "yes" if result.snippet and result.snippet.strip() else "no"
        lines.append(
            (
                f"{idx:02d}. [kept] title={result.title or '(empty)'} | "
                f"url={result.url or '(empty)'} | domain={result.domain or '(empty)'} | "
                f"snippet={snippet_flag} | reason={decision.reason}"
            )
        )
    if not kept:
        lines.append("00. [kept] (none)")
    if diagnostics.ranked_candidates:
        lines.append("Ranked candidates:")
    for idx, candidate in enumerate(diagnostics.ranked_candidates, start=1):
        result = candidate.result
        snippet_flag = "yes" if result.snippet and result.snippet.strip() else "no"
        lines.append(
            (
                f"{idx:02d}. keep reason=ranked | title={result.title or '(empty)'} | "
                f"url={result.url} | domain={result.domain or '(empty)'} | "
                f"snippet={snippet_flag} | entity={candidate.entity_type} | "
                f"name_match={candidate.name_match} | authority={candidate.authority_tier} | "
                f"seo_penalty={candidate.seo_penalty:.2f} | context={candidate.context_strength:.3f} | "
                f"final={candidate.score:.3f} | signals={','.join(candidate.reasons[:8])}"
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
