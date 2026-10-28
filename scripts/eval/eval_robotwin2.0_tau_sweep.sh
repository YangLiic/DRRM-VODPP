#!/usr/bin/env bash
# Local RoboTwin FM tau-schedule sweep.
#
# First 10 args match eval_robotwin2.0_parallel.sh; remaining args are runs:
#   <run_name>:<fm_tau_schedule>
#
# Examples:
#   bash scripts/eval/eval_robotwin2.0_tau_sweep.sh \
#     VODPP click_alarmclock agilex_config /path/to/ckpt \
#     500 1 0 headless true 100 \
#     mean3:0,0.27,0.55,1 uniform3:uniform3
#
#   bash scripts/eval/eval_robotwin2.0_tau_sweep.sh --serial ...  # one GPU, sequential
#
# For Volcano / cloud images use eval_robotwin2.0_tau_sweep_cloud.sh instead.
#
# Env:
#   RESULT_ROOT   where eval results / launcher logs go
#   GPU_ID        serial mode GPU (default 0)
#   SEED_BASE     added to run index for --seed (default = seed arg)
#   EVAL_SH       override evaluator script (cloud wrapper sets this)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${EVAL_SH:-$SCRIPT_DIR/eval_robotwin2.0_parallel.sh}"

SERIAL=0
if [ "${1:-}" = "--serial" ]; then
  SERIAL=1
  shift
fi

if [ "$#" -lt 11 ]; then
  echo "Usage: $0 [--serial] <policy> <task> <task_config> <ckpt> <expert_n> <nproc> <seed> <mode> <save_video> <test_num> <name:schedule> [...]" >&2
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
SCHEDULES=()
for spec in "$@"; do
  if [[ "$spec" != *:* ]]; then
    echo "Bad run spec (want name:schedule): $spec" >&2
    exit 2
  fi
  name="${spec%%:*}"
  sched="${spec#*:}"
  if [ -z "$name" ] || [ -z "$sched" ]; then
    echo "Bad run spec (empty name/schedule): $spec" >&2
    exit 2
  fi
  RUN_NAMES+=("$name")
  SCHEDULES+=("$sched")
done

if [ "${#RUN_NAMES[@]}" -eq 0 ]; then
  echo "At least one <name:schedule> run is required" >&2
  exit 2
fi

RESULT_ROOT="${RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
SEED_BASE="${SEED_BASE:-$SEED_ARG}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$RESULT_ROOT"
LOG_ROOT="$RESULT_ROOT/$TASK_NAME/_launcher_logs"
mkdir -p "$LOG_ROOT"
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
if [ "$SERIAL" -eq 1 ]; then
  MASTER_LOG="$LOG_ROOT/tau_sweep_serial_${STAMP}.log"
  TAG="tau-sweep-serial"
else
  MASTER_LOG="$LOG_ROOT/tau_sweep_${STAMP}.log"
  TAG="tau-sweep"
fi

{
  echo "=============================================="
  echo "[$TAG] stamp=$STAMP"
  echo "[$TAG] policy=$POLICY_NAME task=$TASK_NAME config=$TASK_CONFIG"
  echo "[$TAG] ckpt=$CKPT"
  echo "[$TAG] result_root=$RESULT_ROOT"
  echo "[$TAG] test_num=$TEST_NUM save_video=$SAVE_VIDEO mode=$MODE_ARG"
  echo "[$TAG] runs=${#RUN_NAMES[@]} serial=$SERIAL"
  echo "[$TAG] eval_sh=$EVAL_SH"
  nvidia-smi -L 2>/dev/null || true
  echo "=============================================="
  for i in "${!RUN_NAMES[@]}"; do
    echo "[$TAG] run[$i]=${RUN_NAMES[$i]} schedule=${SCHEDULES[$i]}"
  done
} | tee -a "$MASTER_LOG"

run_one() {
  local index="$1"
  local name="${RUN_NAMES[$index]}"
  local sched="${SCHEDULES[$index]}"
  # Same seed for every schedule under one task (episode_k comparable across runs).
  local seed=$SEED_BASE
  local worker_log="$2"
  local gpu="${3:-}"

  echo "[$TAG] launch name=$name schedule=$sched seed=$seed gpu=${gpu:-$GPU_ID} -> $worker_log" | tee -a "$MASTER_LOG"
  (
    if [ -n "$gpu" ]; then
      export CUDA_VISIBLE_DEVICES="$gpu"
    else
      export CUDA_VISIBLE_DEVICES="$GPU_ID"
    fi
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
    set +e
    run_one "$i" "$worker_log"
    rc=$?
    set -e
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
  echo "fail=$FAIL"
  for i in "${!RUN_NAMES[@]}"; do
    echo "run=${RUN_NAMES[$i]} schedule=${SCHEDULES[$i]}"
  done
} >"$LOG_ROOT/${STAMP}_summary.txt"

exit "$FAIL"
