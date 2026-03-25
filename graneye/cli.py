from __future__ import annotations

import argparse
import json
import sys

from .pipeline import SearchPipelineDiagnostics, resolve_query, resolve_query_with_debug
from .resolution import ResolutionOutput, ScoredCandidate, assess_query_validity
from .search import search_duckduckgo_html, search_duckduckgo_instant_answer


class CLIArgs(argparse.Namespace):
    target_name: str
    target_context: str | None
    debug: bool
    json: bool


def _parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="graneye",
        description="Resolve a likely public profile candidate for a person name.",
    )
    parser.add_argument("target_name", help="Target person name to resolve")
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

    namespace = parser.parse_args(argv, namespace=CLIArgs())
    return namespace


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _render_output(target_name: str, target_context: str | None, output: ResolutionOutput) -> str:
    status = _resolution_status(output)
    if status != "resolved":
        lines = [
            f"Name: {target_name}",
            f"Context: {target_context}" if target_context else "Context: (none)",
            f"Status: {status}",
            f"Confidence: {output.confidence_label}",
            f"Top candidate: {output.normalized_candidate_name or '(none)'}",
            f"Source URL: {output.source_url or '(none)'}",
            f"Reason: {output.no_resolution_reason or output.ambiguity_reason or output.explanation}",
        ]
        return "\n".join(lines)

    lines = [
        f"Name: {target_name}",
        f"Context: {target_context}" if target_context else "Context: (none)",
        "Status: resolved",
        f"Confidence: {output.confidence_label}",
        f"Top candidate: {output.normalized_candidate_name or '(unknown)'}",
        f"Source URL: {output.source_url}",
    ]
    return "\n".join(lines)


def _render_debug_result_output(target_name: str, target_context: str | None, output: ResolutionOutput) -> str:
    status = _resolution_status(output)
    if status != "resolved":
        lines = [
            f"Target name: {target_name}",
            f"Target context: {target_context}" if target_context else "Target context: (none)",
            f"Resolution: {status}",
            f"Reason: {output.no_resolution_reason or output.ambiguity_reason or 'unknown'}",
            f"Confidence: {output.confidence_label}",
            f"Decision reason: {output.explanation}",
        ]
        return "\n".join(lines)

    lines = [
        f"Target name: {target_name}",
        f"Target context: {target_context}" if target_context else "Target context: (none)",
        f"Top candidate: {output.normalized_candidate_name or '(unknown)'}",
        f"Source URL: {output.source_url}",
        f"Display title: {output.source_title or '(not available)'}",
        f"Score: {output.final_score:.3f}",
        f"Same-person probability: {output.same_person_probability:.3f}",
        f"Context match probability: {output.context_match_probability:.3f}",
        f"Confidence: {output.confidence_label}",
        f"Ambiguity detected: {'yes' if output.ambiguity_detected else 'no'}",
        f"Ambiguity reason: {output.ambiguity_reason or '(none)'}",
        f"Resolution path: {output.resolution_path}",
        f"Fetch status: {output.fetch_status}",
        f"Decision reason: {output.explanation}",
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


def _json_payload(
    *,
    target_name: str,
    target_context: str | None,
    query_validity: str,
    output: ResolutionOutput,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_name": target_name,
        "target_context": target_context,
        "query_validity": query_validity,
        "resolution_status": _resolution_status(output),
        "no_resolution_reason": output.no_resolution_reason,
        "ambiguity_reason": output.ambiguity_reason,
        "confidence": output.confidence_label,
        "top_candidate": output.normalized_candidate_name or None,
        "source_url": output.source_url or None,
        "display_title": output.source_title or None,
        "same_person_probability": output.same_person_probability,
        "context_match_probability": output.context_match_probability,
        "entity_type": output.entity_type,
        "decision_reason": output.explanation,
    }
    return payload


def _exit_code(output: ResolutionOutput) -> int:
    status = _resolution_status(output)
    if status == "ambiguous":
        return 1
    if status == "no-resolution":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if _is_blank(args.target_name):
        print("Error: target_name must not be empty.", file=sys.stderr)
        return 3

    target_name = args.target_name.strip()
    context = args.target_context.strip() if args.target_context else None
    query_validity = assess_query_validity(target_name).status

    diagnostics: SearchPipelineDiagnostics | None = None
    try:
        if args.debug:
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
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error: failed to execute pipeline: {exc}", file=sys.stderr)
        return 3

    final_output = _build_final_output(
        query_validity=query_validity,
        resolved=resolved_output,
        ranked=ranked,
    )

    if args.debug and diagnostics is not None:
        print(_render_debug_output(diagnostics))

    if args.json:
        payload = _json_payload(
            target_name=target_name,
            target_context=context,
            query_validity=query_validity,
            output=final_output,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return _exit_code(final_output)

    print(
        _render_debug_result_output(target_name, context, final_output)
        if args.debug
        else _render_output(target_name, context, final_output)
    )
    return _exit_code(final_output)


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
