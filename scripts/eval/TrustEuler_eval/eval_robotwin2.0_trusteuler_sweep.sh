#!/usr/bin/env bash
# RoboTwin eval sweep with TrustEuler Mean-grid sensing (λ=0 / M0).
#
# Flow:
#   1) If any run needs mean* and not --skip-calibrate:
#        wait for trusteuler_calibrate.sh (U100 dense Euler only)
#      If --skip-calibrate: require existing traces under TrustEuler_runs/<task>/
#   2) Serial: for each run, if meanN then sense budget N, then eval
#      Parallel: wait for step-1 first; each worker senses its own budget then evals
#   3) uniform* / explicit knots skip sensing and eval directly
#
# Args: same base 10 args as tau_sweep, then runs:
#   mean3 | mean5 | uniform3:uniform3 | custom:0,0.3,1 | ...
#
# Env: RESULT_ROOT, GPU_ID, SEED_BASE, EVAL_SH, TRUSTEULER_*, FORCE_CALIBRATE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_SH="${EVAL_SH:-$EVAL_DIR/eval_robotwin2.0_parallel.sh}"
CALIBRATE_SH="$SCRIPT_DIR/trusteuler_calibrate.sh"
SENSE_SH="$SCRIPT_DIR/trusteuler_sense_grids.sh"
RESOLVE_PY="$SCRIPT_DIR/resolve_mean_schedule.py"

SERIAL=0
SKIP_CALIBRATE=0
while [ "${1:-}" = "--serial" ] || [ "${1:-}" = "--skip-calibrate" ]; do
  if [ "$1" = "--serial" ]; then
    SERIAL=1
  else
    SKIP_CALIBRATE=1
  fi
  shift
done

if [ "$#" -lt 11 ]; then
  echo "Usage: $0 [--serial] [--skip-calibrate] <policy> <task> <task_config> <ckpt> <expert_n> <nproc> <seed> <mode> <save_video> <test_num> <name[:schedule]> [...]" >&2
  exit 2
fi

POLICY_NAME="$1"
TASK_NAME="$2"
TASK_CONFIG="$3"
CKPT="$4"
EXPERT_DATA_NUM="$5"
NUM_PROCESS="$6"
SEED_ARG="$7"
MODE_ARG="$8"
SAVE_VIDEO="$9"
TEST_NUM="${10}"
shift 10

RUN_NAMES=()
SCHEDULE_TOKENS=()
for spec in "$@"; do
  if [[ "$spec" == *:* ]]; then
    name="${spec%%:*}"
    sched="${spec#*:}"
  else
    name="$spec"
    sched="$spec"
  fi
  if [ -z "$name" ] || [ -z "$sched" ]; then
    echo "Bad run spec: $spec" >&2
    exit 2
  fi
  RUN_NAMES+=("$name")
  SCHEDULE_TOKENS+=("$sched")
done

DRRM_ROOT="${DRRM_ROOT:-$(cd "$EVAL_DIR/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
TRUSTEULER_RUNS_ROOT="${TRUSTEULER_RUNS_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs}"
SEED_BASE="${SEED_BASE:-$SEED_ARG}"
GPU_ID="${GPU_ID:-0}"
RUN_DIR="$TRUSTEULER_RUNS_ROOT/$TASK_NAME"
TRACE_DIR="$RUN_DIR/traces"
MANIFEST="$RUN_DIR/calibration_dataset.json"

MEAN_BUDGETS=()
NEED_MEAN=0
for tok in "${SCHEDULE_TOKENS[@]}"; do
  if [[ "$tok" =~ ^[Mm][Ee][Aa][Nn]([0-9]+)$ ]]; then
    NEED_MEAN=1
    b="${BASH_REMATCH[1]}"
    dup=0
    if [ "${#MEAN_BUDGETS[@]}" -gt 0 ]; then
      for existing in "${MEAN_BUDGETS[@]}"; do
        if [ "$existing" = "$b" ]; then
          dup=1
          break
        fi
      done
    fi
    if [ "$dup" -eq 0 ]; then
      MEAN_BUDGETS+=("$b")
    fi
  fi
done

mkdir -p "$RESULT_ROOT"
LOG_ROOT="$RESULT_ROOT/$TASK_NAME/_launcher_logs"
mkdir -p "$LOG_ROOT"
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
if [ "$SERIAL" -eq 1 ]; then
  MASTER_LOG="$LOG_ROOT/trusteuler_sweep_serial_${STAMP}.log"
  TAG="trusteuler-sweep-serial"
else
  MASTER_LOG="$LOG_ROOT/trusteuler_sweep_${STAMP}.log"
  TAG="trusteuler-sweep"
fi

_pick_resolve_py() {
  if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    echo "$PYTHON"
  elif [ -x "$DRRM_ROOT/simulation/robotwin2.0/.venv/bin/python" ]; then
    echo "$DRRM_ROOT/simulation/robotwin2.0/.venv/bin/python"
  else
    echo "python3"
  fi
}
RESOLVE_PY_BIN="$(_pick_resolve_py)"

mean_grid_for_budget() {
  local b="$1"
  echo "$RUN_DIR/calibration/b${b}/mean_grid.json"
}

resolve_token() {
  local tok="$1"
  if [[ "$tok" =~ ^[Mm][Ee][Aa][Nn]([0-9]+)$ ]]; then
    local b="${BASH_REMATCH[1]}"
    local grid
    grid="$(mean_grid_for_budget "$b")"
    "$RESOLVE_PY_BIN" "$RESOLVE_PY" "$tok" --mean-grid "$grid"
  else
    "$RESOLVE_PY_BIN" "$RESOLVE_PY" "$tok"
  fi
}

sense_budget() {
  local b="$1"
  bash "$SENSE_SH" "$TASK_NAME" --budget "$b"
}

{
  echo "=============================================="
  echo "[$TAG] stamp=$STAMP"
  echo "[$TAG] policy=$POLICY_NAME task=$TASK_NAME config=$TASK_CONFIG"
  echo "[$TAG] ckpt=$CKPT"
  echo "[$TAG] result_root=$RESULT_ROOT"
  echo "[$TAG] trusteuler_runs=$RUN_DIR"
  echo "[$TAG] test_num=$TEST_NUM save_video=$SAVE_VIDEO mode=$MODE_ARG"
  echo "[$TAG] runs=${#RUN_NAMES[@]} serial=$SERIAL need_mean=$NEED_MEAN skip_calibrate=$SKIP_CALIBRATE"
  echo "[$TAG] eval_sh=$EVAL_SH"
  nvidia-smi -L 2>/dev/null || true
  echo "=============================================="
  for i in "${!RUN_NAMES[@]}"; do
    echo "[$TAG] run[$i]=${RUN_NAMES[$i]} token=${SCHEDULE_TOKENS[$i]}"
  done
} | tee -a "$MASTER_LOG"

# ---- Stage 1: U100 dense Euler (shared; wait before any eval) ----
if [ "$NEED_MEAN" -eq 1 ]; then
  if [ "$SKIP_CALIBRATE" -eq 1 ]; then
    if [ ! -f "$MANIFEST" ] || [ -z "$(ls -A "$TRACE_DIR"/*.npz 2>/dev/null || true)" ]; then
      echo "[$TAG] ERROR: --skip-calibrate but missing U100 traces under $TRACE_DIR" | tee -a "$MASTER_LOG"
      exit 1
    fi
    echo "[$TAG] skip U100 collect; reuse $TRACE_DIR" | tee -a "$MASTER_LOG"
  else
    echo "[$TAG] TrustEuler U100 collect (wait before eval) ..." | tee -a "$MASTER_LOG"
    set +e
    bash "$CALIBRATE_SH" "$TASK_NAME" "$CKPT" 2>&1 | tee -a "$MASTER_LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "[$TAG] FAIL TrustEuler U100 collect rc=$rc" | tee -a "$MASTER_LOG"
      exit "$rc"
    fi
  fi
fi

run_one() {
  local index="$1"
  local name="${RUN_NAMES[$index]}"
  local tok="${SCHEDULE_TOKENS[$index]}"
  # Same seed for every schedule of this task so episode k shares env RNG.
  local seed=$SEED_BASE
  local worker_log="$2"
  local gpu="${3:-}"
  local sched=""

  (
    if [ -n "$gpu" ]; then
      export CUDA_VISIBLE_DEVICES="$gpu"
    else
      export CUDA_VISIBLE_DEVICES="$GPU_ID"
    fi

    if [[ "$tok" =~ ^[Mm][Ee][Aa][Nn]([0-9]+)$ ]]; then
      local b="${BASH_REMATCH[1]}"
      echo "[$TAG] sense mean grid budget=$b for name=$name" 
      bash "$SENSE_SH" "$TASK_NAME" --budget "$b"
    fi

    sched="$(resolve_token "$tok")"
    echo "[$TAG] resolved $name: $tok -> $sched"
    echo "[$TAG] launch name=$name schedule=$sched seed=$seed gpu=${gpu:-$GPU_ID}"

    export EVAL_RESULT_ROOT="$RESULT_ROOT"
    export EVAL_RUN_NAME="$name"
    bash "$EVAL_SH" \
      "$POLICY_NAME" "$TASK_NAME" "$TASK_CONFIG" \
      "$CKPT" "$EXPERT_DATA_NUM" "$NUM_PROCESS" "$seed" \
      "$MODE_ARG" "$SAVE_VIDEO" "$TEST_NUM" \
      "$sched"
  ) >"$worker_log" 2>&1
}

FAIL=0
if [ "$SERIAL" -eq 1 ]; then
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"
    worker_log="$LOG_ROOT/${STAMP}_${name}_serial.log"
    echo "[$TAG] serial step name=$name -> $worker_log" | tee -a "$MASTER_LOG"
    set +e
    run_one "$i" "$worker_log"
    rc=$?
    set -e
    # surface resolve/sense lines
    grep -E "\[$TAG\] (sense|resolved|launch)" "$worker_log" | tee -a "$MASTER_LOG" || true
    if ! grep -q "Render Well" "$worker_log"; then
      echo "[$TAG] FAIL name=$name (no Render Well; see $worker_log)" | tee -a "$MASTER_LOG"
      grep -E "Render Error|Error|Traceback|ModuleNotFoundError" "$worker_log" | tail -n 30 | tee -a "$MASTER_LOG" || true
      FAIL=1
      continue
    fi
    if ! grep -q "Success Rate:" "$worker_log" && ! grep -q "Data has been saved to" "$worker_log"; then
      echo "[$TAG] FAIL name=$name (eval did not finish; rc=$rc)" | tee -a "$MASTER_LOG"
      grep -E "Error|Traceback|ModuleNotFoundError|missing " "$worker_log" | tail -n 40 | tee -a "$MASTER_LOG" || true
      tail -n 20 "$worker_log" | tee -a "$MASTER_LOG" || true
      FAIL=1
      continue
    fi
    if [ "$rc" -eq 0 ]; then
      echo "[$TAG] OK   name=$name" | tee -a "$MASTER_LOG"
    else
      echo "[$TAG] FAIL name=$name rc=$rc" | tee -a "$MASTER_LOG"
      FAIL=1
    fi
  done
else
  if [ "${#RUN_NAMES[@]}" -gt 8 ]; then
    echo "[$TAG] warning: ${#RUN_NAMES[@]} runs on presumed 8-GPU host" | tee -a "$MASTER_LOG"
  fi
  echo "[$TAG] parallel eval: each mean* worker senses its own budget" | tee -a "$MASTER_LOG"
  PIDS=()
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"
    worker_log="$LOG_ROOT/${STAMP}_${name}_gpu${i}.log"
    run_one "$i" "$worker_log" "$i" &
    PIDS+=("$!")
  done
  echo "[$TAG] waiting for ${#PIDS[@]} workers: ${PIDS[*]}" | tee -a "$MASTER_LOG"
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    name="${RUN_NAMES[$idx]}"
    if wait "$pid"; then
      echo "[$TAG] OK   name=$name pid=$pid" | tee -a "$MASTER_LOG"
    else
      rc=$?
      echo "[$TAG] FAIL name=$name pid=$pid rc=$rc" | tee -a "$MASTER_LOG"
      FAIL=1
    fi
  done
fi

echo "[$TAG] done fail=$FAIL master_log=$MASTER_LOG" | tee -a "$MASTER_LOG"
{
  echo "stamp=$STAMP"
  echo "serial=$SERIAL"
  echo "ckpt=$CKPT"
  echo "task=$TASK_NAME"
  echo "result_root=$RESULT_ROOT"
  echo "trusteuler_runs=$RUN_DIR"
  echo "fail=$FAIL"
  for i in "${!RUN_NAMES[@]}"; do
    echo "run=${RUN_NAMES[$i]} token=${SCHEDULE_TOKENS[$i]}"
  done
} >"$LOG_ROOT/${STAMP}_trusteuler_summary.txt"

exit "$FAIL"
