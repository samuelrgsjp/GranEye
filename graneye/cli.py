from __future__ import annotations

import argparse
import sys

from .pipeline import SearchPipelineDiagnostics, resolve_query, resolve_query_with_debug
from .resolution import ResolutionOutput
from .search import SearchResult, search_duckduckgo_html, search_duckduckgo_instant_answer


class CLIArgs(argparse.Namespace):
    target_name: str
    target_context: str | None
    debug: bool


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

    namespace = parser.parse_args(argv, namespace=CLIArgs())
    return namespace


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def _render_output(target_name: str, target_context: str | None, output: ResolutionOutput, top: SearchResult) -> str:
    lines = [
        f"Target name: {target_name}",
        f"Target context: {target_context}" if target_context else "Target context: (none)",
        f"Top candidate: {output.normalized_candidate_name}",
        f"Source URL: {output.source_url}",
        f"Display title: {top.title or '(not available)'}",
        f"Score: {output.final_score:.3f}",
        f"Same-person probability: {output.same_person_probability:.3f}",
        f"Context match probability: {output.context_match_probability:.3f}",
        f"Resolution path: {output.resolution_path}",
        f"Fetch status: {output.fetch_status}",
        f"Decision reason: {output.explanation}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if _is_blank(args.target_name):
        print("Error: target_name must not be empty.", file=sys.stderr)
        return 2

    context = args.target_context.strip() if args.target_context else None

    diagnostics: SearchPipelineDiagnostics | None = None
    try:
        if args.debug:
            output, ranked, diagnostics = resolve_query_with_debug(
                args.target_name.strip(),
                context=context,
                html_search=search_duckduckgo_html,
                instant_search=search_duckduckgo_instant_answer,
            )
        else:
            output, ranked = resolve_query(
                args.target_name.strip(),
                context=context,
                html_search=search_duckduckgo_html,
                instant_search=search_duckduckgo_instant_answer,
            )
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Error: failed to execute pipeline: {exc}", file=sys.stderr)
        return 1

    if args.debug and diagnostics is not None:
        print(_render_debug_output(diagnostics))

    if not ranked:
        print(f"No candidates found for '{args.target_name.strip()}'.")
        return 3

    if output is None:
        top = ranked[0]
        print(
            "\n".join(
                [
                    f"Target name: {args.target_name.strip()}",
                    f"Target context: {context}" if context else "Target context: (none)",
                    f"Top candidate (search-only): {top.result.title or top.result.domain}",
                    f"Source URL: {top.result.url}",
                    f"Score: {top.score:.3f}",
                    "Decision reason: content fetch failed; using ranked search evidence only.",
                ]
            )
        )
        return 0

    print(_render_output(args.target_name.strip(), context, output, ranked[0].result))
    return 0


def _render_debug_output(diagnostics: SearchPipelineDiagnostics) -> str:
    lines = [
        "=== Debug: Search pipeline ===",
        f"Query attempts: {' | '.join(diagnostics.query_attempts) if diagnostics.query_attempts else '(none)'}",
        f"Raw results count: {diagnostics.raw_results_count}",
        f"Normalized results count: {diagnostics.normalized_results_count}",
        f"Filtered results count: {diagnostics.filtered_results_count}",
        f"Ranked candidates count: {diagnostics.ranked_candidates_count}",
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
                f"name_match={candidate.name_match} | context={candidate.context_strength:.3f} | "
                f"final={candidate.score:.3f} | signals={','.join(candidate.reasons[:8])}"
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
