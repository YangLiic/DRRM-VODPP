#!/usr/bin/env bash
# One-GPU entry: start the Cosmos policy server, wait for it, run the eval, clean up.
#
#   bash run_eval.sh <task_name> [task_config] [ckpt_setting] [seed] [gpu_id]
#
# Env: COSMOS_RW_CKPT, COSMOS_PORT, COSMOS_RW_NUM_STEPS, COSMOS_RW_GUIDANCE,
#      COSMOS_REPLAN_STEPS, TEST_NUM, EVAL_RUN_NAME, EVAL_RESULT_ROOT
set -euo pipefail

task_name=${1:?task_name required}
task_config=${2:-demo_clean}
ckpt_setting=${3:-cosmos3_edge_robotwin}
seed=${4:-0}
gpu_id=${5:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${COSMOS_PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export COSMOS_PORT="${PORT}"
export COSMOS_RW_PORT="${PORT}"
export COSMOS_HOST="${COSMOS_HOST:-127.0.0.1}"
export EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-${ckpt_setting}}"

LOG_DIR="${COSMOS_RW_LOG_DIR:-/data/vepfs/licheng.yang/logs}"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/cosmos_serve_${PORT}.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cosmos] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[cosmos] starting server -> ${SERVER_LOG}"
bash "${SCRIPT_DIR}/serve.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

port_is_ready() {
  local port="$1"
  grep -q "server listening on ws://.*:${port}/" "${SERVER_LOG}" 2>/dev/null && return 0
  if command -v ss >/dev/null 2>&1; then
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$" && return 0
  fi
  return 1
}

# Cold start loads a 3.9B checkpoint plus the Wan VAE off EFS; allow 20 minutes.
deadline=$((SECONDS + 1200))
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cosmos] server exited early; see ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if port_is_ready "${PORT}"; then
    echo "[cosmos] server ready on :${PORT}"
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[cosmos] timeout waiting for server on :${PORT}; see ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 3
done

bash "${SCRIPT_DIR}/eval.sh" \
  "${task_name}" "${task_config}" "${ckpt_setting}" "${seed}" "${gpu_id}" \
  "127.0.0.1" "${PORT}"
