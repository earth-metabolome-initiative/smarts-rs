#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)
REPO_ROOT=$(
  cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1
  pwd
)
FUZZ_DIR="$REPO_ROOT/fuzz"

DEFAULT_SESSION="smarts-fuzz"
SESSION_NAME="$DEFAULT_SESSION"
ATTACH=true

ALL_HARNESSES=(
  "parse_display_loop"
  "bracket_parse_loop"
  "parse_bracket_atom"
  "recursive_smarts_lowering"
  "validator_match_loop"
)

HARNESSES=()
LIBFUZZER_ARGS=()
USER_SET_LIBFUZZER_ARGS=false
DEFAULT_LIBFUZZER_ARGS=(
  "-timeout=10"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_fuzz_tmux.sh [-s SESSION] [-d] [HARNESS ...] [-- LIBFUZZER_ARGS...]

Starts the selected cargo-fuzz harnesses in tiled tmux panes.
If the target tmux session already exists, it is killed and recreated.

Options:
  -s SESSION  tmux session name (default: smarts-fuzz)
  -d          start detached and do not attach/switch to the session
  -h          show this help

Examples:
  scripts/run_fuzz_tmux.sh
  scripts/run_fuzz_tmux.sh validator_match_loop parse_display_loop
  scripts/run_fuzz_tmux.sh -s fuzzing -- -max_len=4096 -timeout=5

Notes:
  If no libFuzzer args are provided, the script uses per-harness defaults.
  Passing args after -- replaces those defaults completely.
EOF
}

shell_join() {
  local joined=""
  local item
  for item in "$@"; do
    joined+=" $(printf '%q' "$item")"
  done
  printf '%s' "${joined# }"
}

has_harness() {
  local needle=$1
  local harness
  for harness in "${ALL_HARNESSES[@]}"; do
    if [[ "$harness" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

while (($# > 0)); do
  case "$1" in
    -s)
      if (($# < 2)); then
        echo "missing value after -s" >&2
        exit 2
      fi
      SESSION_NAME=$2
      shift 2
      ;;
    -d)
      ATTACH=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      LIBFUZZER_ARGS=("$@")
      USER_SET_LIBFUZZER_ARGS=true
      break
      ;;
    *)
      HARNESSES+=("$1")
      shift
      ;;
  esac
done

if ((${#HARNESSES[@]} == 0)); then
  HARNESSES=("${ALL_HARNESSES[@]}")
fi

if ((${#LIBFUZZER_ARGS[@]} == 0)); then
  LIBFUZZER_ARGS=("${DEFAULT_LIBFUZZER_ARGS[@]}")
fi

for harness in "${HARNESSES[@]}"; do
  if ! has_harness "$harness"; then
    echo "unknown harness: $harness" >&2
    echo "known harnesses: ${ALL_HARNESSES[*]}" >&2
    exit 2
  fi
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is not installed" >&2
  exit 1
fi

if ! (cd "$FUZZ_DIR" && cargo fuzz --help >/dev/null 2>&1); then
  echo "cargo-fuzz is not available. Install it with: cargo install cargo-fuzz" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "replacing existing tmux session: $SESSION_NAME"
  tmux kill-session -t "$SESSION_NAME"
fi

build_pane_command() {
  local harness=$1
  local cargo_cmd=("cargo" "fuzz" "run" "$harness")
  local pane_args=("${LIBFUZZER_ARGS[@]}")

  if [[ "$USER_SET_LIBFUZZER_ARGS" == false ]]; then
    case "$harness" in
      parse_display_loop)
        pane_args=("-timeout=5" "-max_len=256")
        ;;
      bracket_parse_loop)
        pane_args=("-timeout=5" "-max_len=128")
        ;;
      parse_bracket_atom)
        pane_args=("-timeout=5" "-max_len=512")
        ;;
      recursive_smarts_lowering)
        pane_args=("-timeout=5" "-max_len=192")
        ;;
      validator_match_loop)
        pane_args=("-timeout=5" "-max_len=192")
        ;;
    esac
  fi
  if ((${#pane_args[@]} > 0)); then
    cargo_cmd+=("--")
    cargo_cmd+=("${pane_args[@]}")
  fi

  local rendered
  rendered=$(shell_join "${cargo_cmd[@]}")

  printf 'printf "\\033]2;%s\\007"; echo "== %s =="; %s; status=$?; echo; echo "[%s exited with status $status]"; exec bash' \
    "$harness" "$harness" "$rendered" "$harness"
}

first_harness=${HARNESSES[0]}
tmux new-session -d -s "$SESSION_NAME" -n fuzz -c "$FUZZ_DIR" \
  "bash -lc '$(build_pane_command "$first_harness")'"
tmux select-pane -t "$SESSION_NAME":0.0 -T "$first_harness"

for ((index = 1; index < ${#HARNESSES[@]}; index++)); do
  harness=${HARNESSES[$index]}
  tmux split-window -t "$SESSION_NAME":0 -c "$FUZZ_DIR" \
    "bash -lc '$(build_pane_command "$harness")'"
  tmux select-layout -t "$SESSION_NAME":0 tiled >/dev/null
  tmux select-pane -t "$SESSION_NAME":0.$index -T "$harness"
done

tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null
tmux set-window-option -t "$SESSION_NAME":0 pane-border-status top >/dev/null
tmux select-layout -t "$SESSION_NAME":0 tiled >/dev/null

if [[ "$ATTACH" == true ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION_NAME"
  else
    tmux attach-session -t "$SESSION_NAME"
  fi
else
  echo "started tmux session: $SESSION_NAME"
fi
