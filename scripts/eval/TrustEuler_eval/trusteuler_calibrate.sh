#!/usr/bin/env bash
# TrustEuler dense-trace collection for one RoboTwin task/ckpt.
#
# ONLY:
#   prepare configs → build-dataset → collect-vodpp (U100 dense Euler)
# Does NOT freeze mean/scmean inference grids (that happens at eval time).
#
# Artifacts:
#   $TRUSTEULER_RUNS_ROOT/<task>/
#     configs/{dataset,vodpp}.json
#     calibration_dataset.json
#     traces/*.npz
#
# Usage:
#   bash trusteuler_calibrate.sh <task> <checkpoint>
#
# Env:
#   TRUSTEULER_ROOT / TRUSTEULER_RUNS_ROOT / ROBOTWIN_DATASETS / DRRM_ROOT
#   PYTHON / TE_PYTHON / EPISODE_COUNT / DENSE_STEPS / USE_EMA / DEVICE
#   FORCE_CALIBRATE=1  redo even if traces already look complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRRM_ROOT="${DRRM_ROOT:-$(cd "$EVAL_DIR/../.." && pwd)}"
TRUSTEULER_ROOT="${TRUSTEULER_ROOT:-/data/vepfs/licheng.yang/TrustEuler}"
TRUSTEULER_RUNS_ROOT="${TRUSTEULER_RUNS_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs}"
ROBOTWIN_DATASETS="${ROBOTWIN_DATASETS:-$DRRM_ROOT/simulation/robotwin2.0/datasets}"
EPISODE_COUNT="${EPISODE_COUNT:-8}"
DENSE_STEPS="${DENSE_STEPS:-100}"
USE_EMA="${USE_EMA:-false}"
DEVICE="${DEVICE:-cuda:0}"
FORCE_CALIBRATE="${FORCE_CALIBRATE:-0}"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <task> <checkpoint>" >&2
  exit 2
fi

TASK="$1"
CKPT="$2"

RUN_DIR="$TRUSTEULER_RUNS_ROOT/$TASK"
CFG_DIR="$RUN_DIR/configs"
MANIFEST="$RUN_DIR/calibration_dataset.json"
TRACE_DIR="$RUN_DIR/traces"

_pick_python() {
  if [ -n "${TE_PYTHON:-}" ] && [ -x "${TE_PYTHON}" ]; then
    echo "$TE_PYTHON"
    return 0
  fi
  if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    echo "$PYTHON"
    return 0
  fi
  local cand
  for cand in \
    "$DRRM_ROOT/simulation/robotwin2.0/.venv/bin/python" \
    "$TRUSTEULER_ROOT/.venv/bin/python" \
    "$(command -v python3 || true)"
  do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

TE_PY="$(_pick_python)" || {
  echo "[trusteuler-calibrate] ERROR: no python found" >&2
  exit 1
}

export PYTHONPATH="${TRUSTEULER_ROOT}/src:${DRRM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

mkdir -p "$RUN_DIR" "$CFG_DIR" "$TRACE_DIR"

echo "=============================================="
echo "[trusteuler-calibrate] task=$TASK"
echo "[trusteuler-calibrate] ckpt=$CKPT"
echo "[trusteuler-calibrate] run_dir=$RUN_DIR"
echo "[trusteuler-calibrate] python=$TE_PY"
echo "[trusteuler-calibrate] dense_steps=$DENSE_STEPS (U${DENSE_STEPS} only; no mean-grid)"
echo "=============================================="

_traces_complete() {
  [ -f "$MANIFEST" ] || return 1
  "$TE_PY" - "$MANIFEST" "$TRACE_DIR" <<'PY'
import json, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
trace_dir = Path(sys.argv[2])
want = int(manifest.get("dataset", {}).get("chunk_count", 0))
have = len(list(trace_dir.glob("*.npz")))
sys.exit(0 if want > 0 and have >= want else 1)
PY
}

if [ "$FORCE_CALIBRATE" != "1" ] && _traces_complete; then
  echo "[trusteuler-calibrate] reuse existing U${DENSE_STEPS} traces under $TRACE_DIR"
  exit 0
fi

EMA_FLAG=()
if [ "$USE_EMA" = "true" ] || [ "$USE_EMA" = "1" ]; then
  EMA_FLAG=(--use-ema)
fi

"$TE_PY" "$SCRIPT_DIR/prepare_trusteuler_configs.py" \
  --task "$TASK" \
  --checkpoint "$CKPT" \
  --dataset-root "$ROBOTWIN_DATASETS/$TASK" \
  --drrm-root "$DRRM_ROOT" \
  --output-dir "$CFG_DIR" \
  --episode-count "$EPISODE_COUNT" \
  --dense-steps "$DENSE_STEPS" \
  "${EMA_FLAG[@]}"

echo "[trusteuler-calibrate] build-dataset"
"$TE_PY" -m trusteuler build-dataset \
  --config "$CFG_DIR/dataset.json" \
  --output "$MANIFEST"

echo "[trusteuler-calibrate] collect-vodpp (dense U${DENSE_STEPS} Euler)"
cd "$DRRM_ROOT"
"$TE_PY" -m trusteuler collect-vodpp \
  --manifest "$MANIFEST" \
  --backend-config "$CFG_DIR/vodpp.json" \
  --output-dir "$TRACE_DIR" \
  --device "$DEVICE"

echo "[trusteuler-calibrate] done (traces only) -> $TRACE_DIR"
