#!/usr/bin/env bash
# Freeze Mean (λ=0 / M0) inference grids from existing U100 traces.
#
# Does NOT collect traces. Expects:
#   $TRUSTEULER_RUNS_ROOT/<task>/calibration_dataset.json
#   $TRUSTEULER_RUNS_ROOT/<task>/traces/*.npz
#
# Writes per-budget grids (safe for parallel workers):
#   $TRUSTEULER_RUNS_ROOT/<task>/calibration/b<N>/mean_grid.json
#
# Usage:
#   bash trusteuler_sense_grids.sh <task> --budget 3 [--budget 5 ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRRM_ROOT="${DRRM_ROOT:-$(cd "$EVAL_DIR/../.." && pwd)}"
TRUSTEULER_ROOT="${TRUSTEULER_ROOT:-/data/vepfs/licheng.yang/TrustEuler}"
TRUSTEULER_RUNS_ROOT="${TRUSTEULER_RUNS_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs}"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <task> --budget N [--budget M ...]" >&2
  exit 2
fi

TASK="$1"
shift

BUDGETS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --budget)
      [ "$#" -ge 2 ] || { echo "missing value for --budget" >&2; exit 2; }
      BUDGETS+=("$2")
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ "${#BUDGETS[@]}" -eq 0 ]; then
  echo "at least one --budget is required" >&2
  exit 2
fi

RUN_DIR="$TRUSTEULER_RUNS_ROOT/$TASK"
MANIFEST="$RUN_DIR/calibration_dataset.json"
TRACE_DIR="$RUN_DIR/traces"

if [ ! -f "$MANIFEST" ]; then
  echo "[trusteuler-sense] ERROR: missing manifest $MANIFEST" >&2
  exit 1
fi
if [ ! -d "$TRACE_DIR" ] || [ -z "$(ls -A "$TRACE_DIR"/*.npz 2>/dev/null || true)" ]; then
  echo "[trusteuler-sense] ERROR: no traces under $TRACE_DIR" >&2
  exit 1
fi

_pick_python() {
  if [ -n "${TE_PYTHON:-}" ] && [ -x "${TE_PYTHON}" ]; then
    echo "$TE_PYTHON"; return 0
  fi
  if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    echo "$PYTHON"; return 0
  fi
  local cand
  for cand in \
    "$DRRM_ROOT/simulation/robotwin2.0/.venv/bin/python" \
    "$TRUSTEULER_ROOT/.venv/bin/python" \
    "$(command -v python3 || true)"
  do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

TE_PY="$(_pick_python)" || {
  echo "[trusteuler-sense] ERROR: no python found" >&2
  exit 1
}

export PYTHONPATH="${TRUSTEULER_ROOT}/src:${DRRM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for b in "${BUDGETS[@]}"; do
  out_dir="$RUN_DIR/calibration/b${b}"
  mkdir -p "$out_dir"
  grid="$out_dir/mean_grid.json"
  if [ -f "$grid" ]; then
    if "$TE_PY" - "$grid" "$b" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if str(sys.argv[2]) in (obj.get("schedules") or {}) else 1)
PY
    then
      echo "[trusteuler-sense] reuse budget=$b -> $grid"
      continue
    fi
  fi
  echo "[trusteuler-sense] calibrate budget=$b (lambda-seq=0) -> $out_dir"
  "$TE_PY" -m trusteuler calibrate \
    --manifest "$MANIFEST" \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$out_dir" \
    --budget "$b" \
    --lambda-seq 0
done

echo "[trusteuler-sense] done task=$TASK budgets=${BUDGETS[*]}"
