#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/validation_reports"
mkdir -p "$OUT_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FULL_LOG="$OUT_DIR/validation_${TIMESTAMP}.log"
SUMMARY_FILE="$OUT_DIR/validation_${TIMESTAMP}_summary.txt"
FOCUS_FILE="$OUT_DIR/validation_${TIMESTAMP}_focus.txt"
LATEST_LOG="$OUT_DIR/validation_latest.log"
LATEST_SUMMARY="$OUT_DIR/validation_latest_summary.txt"
LATEST_FOCUS="$OUT_DIR/validation_latest_focus.txt"

# case format: group|name|context
CASES=(
  "strong_resolve|Jensen Huang|NVIDIA CEO"
  "strong_resolve|Jensen Huang|NVIDIA founder CEO"
  "strong_resolve|Satya Nadella|Microsoft executive"
  "public_figures|Pope Francis|vatican"
  "public_figures|Lionel Messi|football argentina"
  "public_figures|MrBeast|youtube"
  "public_figures|Taylor Swift|music singer usa"
  "common_name_no_resolution|Carlos Pérez|Cybersecurity Spain"
  "common_name_no_resolution|David Martínez|Software Engineer Madrid"
  "invalid_queries|123456|"
  "invalid_queries|a|"
)

FOCUS_CASES=(
  "Jensen Huang | NVIDIA CEO"
  "Satya Nadella | Microsoft executive"
  "MrBeast | youtube"
  "Carlos Pérez | Cybersecurity Spain"
  "David Martínez | Software Engineer Madrid"
  "123456 | <none>"
  "a | <none>"
)

print_divider() {
  printf '\n%s\n' "--------------------------------------------------------------------------------"
}

extract_field() {
  local block="$1"
  local label="$2"
  local value
  value="$(printf '%s\n' "$block" | awk -F': ' -v key="$label" '$1==key {print substr($0, length(key)+3)}' | head -n 1)"
  if [[ -z "$value" ]]; then
    value="(none)"
  fi
  printf '%s\n' "$value"
}

write_summary_block() {
  local group="$1"
  local name="$2"
  local context="$3"
  local block="$4"

  {
    printf 'CASE: [%s] %s | %s\n' "$group" "$name" "${context:-<none>}"
    printf 'Query validity: %s\n' "$(extract_field "$block" "Query validity")"
    printf 'Top candidate: %s\n' "$(extract_field "$block" "Top candidate")"
    printf 'Source URL: %s\n' "$(extract_field "$block" "Source URL")"
    printf 'Display title: %s\n' "$(extract_field "$block" "Display title")"
    printf 'Resolution: %s\n' "$(extract_field "$block" "Resolution")"
    printf 'Reason: %s\n' "$(extract_field "$block" "Reason")"
    printf 'Confidence: %s\n' "$(extract_field "$block" "Confidence")"
    printf 'Ambiguity reason: %s\n' "$(extract_field "$block" "Ambiguity reason")"
    printf 'Decision reason: %s\n' "$(extract_field "$block" "Decision reason")"
    printf '\n'
  } >> "$SUMMARY_FILE"
}

run_case() {
  local group="$1"
  local name="$2"
  local context="$3"

  print_divider | tee -a "$FULL_LOG"
  printf 'CASE [%s]: %s | %s\n' "$group" "$name" "${context:-<none>}" | tee -a "$FULL_LOG"
  print_divider | tee -a "$FULL_LOG"

  local cmd=(python -m graneye "$name")
  if [[ -n "$context" ]]; then
    cmd+=("$context")
  fi
  cmd+=(--debug)

  printf 'Command: PYTHONPATH=%q ' "$ROOT_DIR" | tee -a "$FULL_LOG"
  printf '%q ' "${cmd[@]}" | tee -a "$FULL_LOG"
  printf '\n' | tee -a "$FULL_LOG"

  local case_output
  local exit_code=0
  set +e
  case_output="$(PYTHONPATH="$ROOT_DIR" "${cmd[@]}" 2>&1)"
  exit_code=$?
  set -e
  case_output+=$'\n'"CLI exit code: $exit_code"
  printf '%s\n' "$case_output" | tee -a "$FULL_LOG"
  write_summary_block "$group" "$name" "$context" "$case_output"
}

build_focus_file() {
  : > "$FOCUS_FILE"
  for case_key in "${FOCUS_CASES[@]}"; do
    awk -v case_key="$case_key" '
      $0 ~ /^CASE:/ {capture = index($0, case_key) > 0}
      capture {print}
      capture && NF==0 {capture = 0; print ""}
    ' "$SUMMARY_FILE" >> "$FOCUS_FILE"
  done
}

main() {
  : > "$FULL_LOG"
  : > "$SUMMARY_FILE"

  {
    printf 'GranEye validation run\n'
    printf 'Timestamp (UTC): %s\n' "$TIMESTAMP"
    printf 'Repository root: %s\n' "$ROOT_DIR"
    printf '\n'
  } | tee -a "$FULL_LOG"

  local entry group name context
  for entry in "${CASES[@]}"; do
    IFS='|' read -r group name context <<< "$entry"
    run_case "$group" "$name" "$context"
  done

  build_focus_file

  cp "$FULL_LOG" "$LATEST_LOG"
  cp "$SUMMARY_FILE" "$LATEST_SUMMARY"
  cp "$FOCUS_FILE" "$LATEST_FOCUS"

  print_divider
  echo "Validation completed."
  echo "Full log:    $FULL_LOG"
  echo "Summary:     $SUMMARY_FILE"
  echo "Focus file:  $FOCUS_FILE"
  echo "Latest log:  $LATEST_LOG"
  echo "Latest summary: $LATEST_SUMMARY"
  echo "Latest focus:   $LATEST_FOCUS"
}

main "$@"
