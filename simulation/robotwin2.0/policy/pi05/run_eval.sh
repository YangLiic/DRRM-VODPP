#!/usr/bin/env bash
# One-GPU entry (local L4 / Volcano): start openpi server in background, then RoboTwin eval.
#
# Usage:
#   bash run_eval.sh <task_name> [task_config] [ckpt_setting] [seed] [gpu_id]
#
# Env overrides:
#   PI05_CKPT, OPENPI_ROOT, PI05_PORT, PI05_REPLAN_STEPS, PI05_PROMPT_MODE, INSTRUCTION_TYPE
#   PI05_FM_TAU_SCHEDULE / PI05_NUM_STEPS  (passed through to serve.sh)
#   EVAL_RUN_NAME  (result subdir; default = ckpt_setting)
set -euo pipefail

task_name=${1:?task_name required}
task_config=${2:-agilex_config}
ckpt_setting=${3:-robotwin2.0_agilex_3task_25000}
seed=${4:-0}
gpu_id=${5:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PI05_PORT:-8000}"
OPENPI_ROOT="${OPENPI_ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang/openpi}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PI05_PORT="${PORT}"
export PI05_HOST="${PI05_HOST:-127.0.0.1}"
export EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-${ckpt_setting}}"

LOG_DIR="${OPENPI_ROOT}/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/pi05_serve_${PORT}.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[pi05] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[pi05] starting server -> ${SERVER_LOG}"
if [[ -n "${PI05_FM_TAU_SCHEDULE:-}" ]]; then
  echo "[pi05] PI05_FM_TAU_SCHEDULE=${PI05_FM_TAU_SCHEDULE}"
elif [[ -n "${PI05_NUM_STEPS:-}" ]]; then
  echo "[pi05] PI05_NUM_STEPS=${PI05_NUM_STEPS}"
fi
bash "${SCRIPT_DIR}/serve.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Wait until TCP port is listening. Prefer serve-log marker (portable on Volcano
# images without `ss`); fall back to ss / python connect.
port_is_ready() {
  local port="$1"
  if grep -qE "server listening on .*:${port}\\b" "${SERVER_LOG}" 2>/dev/null; then
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
      return 0
    fi
  fi
  return 1
}

deadline=$((SECONDS + 600))
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[pi05] server exited early; see ${SERVER_LOG}" >&2
    tail -n 40 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if port_is_ready "${PORT}"; then
    echo "[pi05] server ready on :${PORT}"
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[pi05] timeout waiting for server on :${PORT}; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

bash "${SCRIPT_DIR}/eval.sh" \
  "${task_name}" "${task_config}" "${ckpt_setting}" "${seed}" "${gpu_id}" \
  "127.0.0.1" "${PORT}"
