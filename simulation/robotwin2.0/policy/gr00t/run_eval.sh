#!/usr/bin/env bash
# One-GPU entry (local L4 / Volcano): start GR00T server in background, then RoboTwin eval.
#
# Usage:
#   bash run_eval.sh <task_name> [task_config] [ckpt_setting] [seed] [gpu_id]
#
# Env overrides:
#   GR00T_CKPT, GR00T_ROOT, GR00T_PORT, GR00T_REPLAN_STEPS, GR00T_PROMPT_MODE, INSTRUCTION_TYPE
#   EVAL_RUN_NAME  (result subdir; default = ckpt_setting)
set -euo pipefail

task_name=${1:?task_name required}
task_config=${2:-agilex_config}
ckpt_setting=${3:-n17_robotwin50_clean_bgr2rgb_h32_40000}
seed=${4:-0}
gpu_id=${5:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${GR00T_PORT:-5555}"
GR00T_ROOT="${GR00T_ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export GR00T_PORT="${PORT}"
export GR00T_HOST="${GR00T_HOST:-127.0.0.1}"
export EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
export EVAL_RUN_NAME="${EVAL_RUN_NAME:-${ckpt_setting}}"

LOG_DIR="${GR00T_ROOT}/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/gr00t_serve_${PORT}.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[gr00t] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[gr00t] starting server -> ${SERVER_LOG}"
bash "${SCRIPT_DIR}/serve.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

port_is_ready() {
  local port="$1"
  if grep -qE "Server ready|listening on .*:${port}\\b|listening on ${port}\\b" "${SERVER_LOG}" 2>/dev/null; then
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
      return 0
    fi
  fi
  # ss is absent on the 8-GPU workers, and the server does not print a ready line, so
  # without this the probe can only time out there.  A plain TCP connect needs no tools.
  if "${PROBE_PY:-python3}" - "${port}" <<'PYPROBE' >/dev/null 2>&1; then
import socket, sys
s = socket.socket(); s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PYPROBE
    return 0
  fi
  return 1
}

# N1.7 load (Cosmos backbone + 3B weights) can take several minutes on L4.
deadline=$((SECONDS + 900))
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[gr00t] server exited early; see ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if port_is_ready "${PORT}"; then
    echo "[gr00t] server ready on :${PORT}"
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "[gr00t] timeout waiting for server on :${PORT}; see ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

bash "${SCRIPT_DIR}/eval.sh" \
  "${task_name}" "${task_config}" "${ckpt_setting}" "${seed}" "${gpu_id}" \
  "127.0.0.1" "${PORT}"
