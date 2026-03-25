from __future__ import annotations

import argparse
import json
import sys

from .pipeline import SearchPipelineDiagnostics, resolve_query, resolve_query_with_debug
from .resolution import ResolutionOutput, assess_query_validity
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
    if output.no_resolution:
        lines = [
            f"Name: {target_name}",
            f"Context: {target_context}" if target_context else "Context: (none)",
            f"Status: {'ambiguous' if output.ambiguity_detected else 'no-resolution'}",
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
    if output.no_resolution:
        lines = [
            f"Target name: {target_name}",
            f"Target context: {target_context}" if target_context else "Target context: (none)",
            "Resolution: no-resolution",
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
        return "resolved"
    if output.ambiguity_detected:
        return "ambiguous"
    if output.no_resolution:
        return "no-resolution"
    return "resolved"


def _json_payload(
    *,
    target_name: str,
    target_context: str | None,
    query_validity: str,
    output: ResolutionOutput | None,
    has_ranked_candidates: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_name": target_name,
        "target_context": target_context,
        "query_validity": query_validity,
        "resolution_status": _resolution_status(output),
        "no_resolution_reason": output.no_resolution_reason if output else None,
        "ambiguity_reason": output.ambiguity_reason if output else None,
        "confidence": output.confidence_label if output else "low",
        "top_candidate": output.normalized_candidate_name if output else None,
        "source_url": output.source_url if output else None,
        "display_title": output.source_title if output else None,
        "same_person_probability": output.same_person_probability if output else None,
        "context_match_probability": output.context_match_probability if output else None,
        "entity_type": output.entity_type if output else None,
        "decision_reason": output.explanation if output else ("content fetch failed; using ranked search evidence only." if has_ranked_candidates else "no candidates"),
    }
    return payload


def _exit_code(output: ResolutionOutput | None, ranked_count: int) -> int:
    if ranked_count == 0:
        return 2
    if output is None:
        return 0
    if output.ambiguity_detected:
        return 1
    if output.no_resolution:
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
            output, ranked, diagnostics = resolve_query_with_debug(
                target_name,
                context=context,
                html_search=search_duckduckgo_html,
                instant_search=search_duckduckgo_instant_answer,
            )
        else:
            output, ranked = resolve_query(
                target_name,
                context=context,
                html_search=search_duckduckgo_html,
                instant_search=search_duckduckgo_instant_answer,
            )
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error: failed to execute pipeline: {exc}", file=sys.stderr)
        return 3

    if args.debug and diagnostics is not None:
        print(_render_debug_output(diagnostics))

    if args.json:
        payload = _json_payload(
            target_name=target_name,
            target_context=context,
            query_validity=query_validity,
            output=output,
            has_ranked_candidates=bool(ranked),
        )
        if output is None and ranked:
            top = ranked[0]
            payload["top_candidate"] = top.result.title or top.result.domain
            payload["source_url"] = top.result.url
            payload["display_title"] = top.result.title or None
            payload["same_person_probability"] = top.score
            payload["context_match_probability"] = top.context_strength
            payload["entity_type"] = top.entity_type
        print(json.dumps(payload, ensure_ascii=False))
        return _exit_code(output, len(ranked))

    if not ranked:
        print(
            "\n".join(
                [
                    f"Name: {target_name}",
                    f"Context: {context}" if context else "Context: (none)",
                    "Status: no-resolution",
                    "Confidence: low",
                    "Top candidate: (none)",
                    "Source URL: (none)",
                    "Reason: no candidates found",
                ]
            )
        )
        return 2

    if output is None:
        top = ranked[0]
        print(
            "\n".join(
                [
                    f"Name: {target_name}",
                    f"Context: {context}" if context else "Context: (none)",
                    "Status: resolved",
                    "Confidence: low",
                    f"Top candidate: {top.result.title or top.result.domain}",
                    f"Source URL: {top.result.url}",
                ]
            )
        )
        return 0

    print(_render_debug_result_output(target_name, context, output) if args.debug else _render_output(target_name, context, output))
    return _exit_code(output, len(ranked))


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
