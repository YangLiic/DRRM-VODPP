#!/usr/bin/env bash
# RoboTwin FM tau-schedule sweep for pi0.5.
#
# Default: parallel — one GPU per schedule, each with its own openpi server
# on a distinct port (BASE_PORT + gpu_index).
#   --serial: one GPU, sequential (start server → eval → tear down).
#
# Usage:
#   bash eval_pi05_tau_sweep.sh \
#     <task> <task_config> <ckpt_setting> <seed> <test_num> \
#     <name:schedule> [...]
#
#   bash eval_pi05_tau_sweep.sh --serial ...
#
# Examples:
#   bash eval_pi05_tau_sweep.sh adjust_bottle agilex_config 3task_35000 0 100 \
#     uniform10:uniform10 sharp3:1,0.581,0.232,0
#
# Schedule tokens are pi0.5 order (1→0). See ../fm_schedule.py.
# For Volcano / cloud images use eval_pi05_tau_sweep_cloud.sh instead.
#
# Seed logic (same as scripts/eval/TrustEuler_eval + eval_robotwin2.0_tau_sweep):
#   All schedules under one task share SEED_BASE (= seed arg, default 0).
#   RoboTwin maps seed -> st_seed = 100000*(1+seed); episode_k is comparable.
#   Do NOT offset seed by run/gpu index (legacy serial mistake).
#
# Env:
#   RESULT_ROOT / EVAL_RESULT_ROOT   result root
#   SEED_BASE                        override shared seed (default = seed arg)
#   GPU_ID                           serial-mode CUDA device (default 0)
#   PI05_PORT                        serial-mode server port (default 8000)
#   PI05_PORT_BASE                   parallel-mode base port (default 8000);
#                                    worker i uses PORT_BASE+i
#   PI05_CKPT / OPENPI_ROOT / ...    same as run_eval.sh
#   RUN_EVAL_SH                      override single-run entry (default ../run_eval.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESOLVE_PY="${PI05_DIR}/fm_schedule.py"
RUN_EVAL_SH="${RUN_EVAL_SH:-${PI05_DIR}/run_eval.sh}"

SERIAL=0
if [ "${1:-}" = "--serial" ]; then
  SERIAL=1
  shift
fi

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 [--serial] <task> <task_config> <ckpt_setting> <seed> <test_num> <name:schedule> [...]" >&2
  exit 2
fi

TASK_NAME="$1"
TASK_CONFIG="$2"
CKPT_SETTING="$3"
SEED_ARG="$4"
TEST_NUM="$5"
shift 5

# Same as TrustEuler / VODPP RoboTwin sweeps: one seed for every schedule.
SEED_BASE="${SEED_BASE:-$SEED_ARG}"

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
  # Validate pi0.5 schedule early (fail before loading the model).
  if ! python3 "${RESOLVE_PY}" "${sched}" >/dev/null; then
    echo "Bad schedule for run=${name}: ${sched}" >&2
    exit 2
  fi
  RUN_NAMES+=("$name")
  SCHEDULES+=("$sched")
done

if [ "${#RUN_NAMES[@]}" -eq 0 ]; then
  echo "At least one <name:schedule> run is required" >&2
  exit 2
fi

RESULT_ROOT="${RESULT_ROOT:-${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}}"
GPU_ID="${GPU_ID:-0}"
PORT_SERIAL="${PI05_PORT:-8000}"
PORT_BASE="${PI05_PORT_BASE:-8000}"

mkdir -p "$RESULT_ROOT"
LOG_ROOT="$RESULT_ROOT/$TASK_NAME/_launcher_logs"
mkdir -p "$LOG_ROOT"
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
if [ "$SERIAL" -eq 1 ]; then
  MASTER_LOG="$LOG_ROOT/pi05_tau_sweep_serial_${STAMP}.log"
  TAG="pi05-tau-sweep-serial"
else
  MASTER_LOG="$LOG_ROOT/pi05_tau_sweep_${STAMP}.log"
  TAG="pi05-tau-sweep"
fi

{
  echo "=============================================="
  echo "[$TAG] stamp=$STAMP"
  echo "[$TAG] task=$TASK_NAME config=$TASK_CONFIG ckpt_setting=$CKPT_SETTING"
  echo "[$TAG] seed=$SEED_BASE (shared; st_seed=$((100000 * (1 + SEED_BASE)))) test_num=$TEST_NUM serial=$SERIAL"
  if [ "$SERIAL" -eq 1 ]; then
    echo "[$TAG] gpu=$GPU_ID port=$PORT_SERIAL"
  else
    echo "[$TAG] port_base=$PORT_BASE (worker i -> gpu=i port=base+i)"
  fi
  echo "[$TAG] result_root=$RESULT_ROOT"
  echo "[$TAG] runs=${#RUN_NAMES[@]}"
  echo "[$TAG] run_eval_sh=$RUN_EVAL_SH"
  echo "[$TAG] PI05_CKPT=${PI05_CKPT:-unset}"
  nvidia-smi -L 2>/dev/null || true
  echo "=============================================="
  for i in "${!RUN_NAMES[@]}"; do
    resolved="$(python3 "${RESOLVE_PY}" "${SCHEDULES[$i]}")"
    echo "[$TAG] run[$i]=${RUN_NAMES[$i]} schedule=${SCHEDULES[$i]} -> ${resolved}"
  done
} | tee -a "$MASTER_LOG"

run_one() {
  local index="$1"
  local name="${RUN_NAMES[$index]}"
  local sched="${SCHEDULES[$index]}"
  local worker_log="$2"
  local gpu="$3"
  local port="$4"

  # Same seed for every schedule of this task so episode k shares env RNG.
  local seed=$SEED_BASE
  echo "[$TAG] launch name=$name schedule=$sched seed=$seed gpu=$gpu port=$port -> $worker_log" \
    | tee -a "$MASTER_LOG"
  (
    # Let run_eval.sh own CUDA_VISIBLE_DEVICES via gpu_id (avoid double-filter).
    export KRAKEN_EGL_DEVICE="$gpu"
    export EVAL_RESULT_ROOT="$RESULT_ROOT"
    export EVAL_RUN_NAME="$name"
    export PI05_FM_TAU_SCHEDULE="$sched"
    export PI05_PORT="$port"
    export TEST_NUM="$TEST_NUM"
    unset PI05_NUM_STEPS
    bash "$RUN_EVAL_SH" \
      "$TASK_NAME" "$TASK_CONFIG" "$CKPT_SETTING" \
      "$seed" "$gpu"
  ) >"$worker_log" 2>&1
}

check_worker() {
  local name="$1"
  local worker_log="$2"
  local rc="$3"
  if grep -q "Data has been saved to" "$worker_log"; then
    if [ "$rc" -eq 0 ]; then
      echo "[$TAG] OK   name=$name" | tee -a "$MASTER_LOG"
      return 0
    fi
    echo "[$TAG] FAIL name=$name (saved but rc=$rc)" | tee -a "$MASTER_LOG"
    return 1
  fi
  echo "[$TAG] FAIL name=$name (eval did not finish; rc=$rc)" | tee -a "$MASTER_LOG"
  grep -E "Error|Traceback|ModuleNotFoundError|ERROR:|timeout|exited early" "$worker_log" \
    | tail -n 40 | tee -a "$MASTER_LOG" || true
  tail -n 20 "$worker_log" | tee -a "$MASTER_LOG" || true
  return 1
}

FAIL=0
if [ "$SERIAL" -eq 1 ]; then
  for i in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$i]}"
    worker_log="$LOG_ROOT/${STAMP}_${name}_serial.log"
    set +e
    run_one "$i" "$worker_log" "$GPU_ID" "$PORT_SERIAL"
    rc=$?
    set -e
    if ! check_worker "$name" "$worker_log" "$rc"; then
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
    port=$((PORT_BASE + i))
    run_one "$i" "$worker_log" "$i" "$port" &
    PIDS+=("$!")
  done
  echo "[$TAG] waiting for ${#PIDS[@]} workers: ${PIDS[*]}" | tee -a "$MASTER_LOG"
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    name="${RUN_NAMES[$idx]}"
    worker_log="$LOG_ROOT/${STAMP}_${name}_gpu${idx}.log"
    set +e
    wait "$pid"
    rc=$?
    set -e
    if ! check_worker "$name" "$worker_log" "$rc"; then
      FAIL=1
    fi
  done
fi

echo "[$TAG] done fail=$FAIL master_log=$MASTER_LOG" | tee -a "$MASTER_LOG"
{
  echo "stamp=$STAMP"
  echo "serial=$SERIAL"
  echo "task=$TASK_NAME"
  echo "ckpt_setting=$CKPT_SETTING"
  echo "seed=$SEED_BASE"
  echo "result_root=$RESULT_ROOT"
  echo "fail=$FAIL"
  for i in "${!RUN_NAMES[@]}"; do
    echo "run=${RUN_NAMES[$i]} schedule=${SCHEDULES[$i]}"
  done
} >"$LOG_ROOT/${STAMP}_summary.txt"

exit "$FAIL"
