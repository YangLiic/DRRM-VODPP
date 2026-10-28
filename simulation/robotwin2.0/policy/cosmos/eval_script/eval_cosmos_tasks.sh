#!/usr/bin/env bash
# Serial multi-task eval against ONE policy server.
#
# run_eval.sh starts and stops a server per task, and a cold start is minutes of
# reading a 3.9B checkpoint plus the Wan VAE off EFS. For a sweep over several
# tasks that is the dominant cost, so this script pays it once.
#
#   bash eval_cosmos_tasks.sh [task_config] [ckpt_setting] [seed] [gpu_id]
#
# Env:
#   COSMOS_TASKS          space-separated task list (default: the 4-task smoke set)
#   TEST_NUM              episodes per task (default 3)
#   COSMOS_PORT           websocket port (default 8000)
#   COSMOS_RW_NUM_STEPS   denoise steps (default 30)
#   EVAL_RESULT_ROOT      result root
set -euo pipefail

task_config=${1:-demo_clean}
ckpt_setting=${2:-cosmos3_edge_robotwin}
seed=${3:-0}
gpu_id=${4:-0}

# Four bimanual tasks with 400-500 step limits, i.e. ~25-32 inference calls each.
# All are in the official RoboTwin 2.0 aloha-agilex release the finetune trained on.
TASKS=${COSMOS_TASKS:-"beat_block_hammer click_bell place_empty_cup adjust_bottle"}
TEST_NUM=${TEST_NUM:-3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${COSMOS_PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export COSMOS_PORT="${PORT}"
export COSMOS_RW_PORT="${PORT}"
export COSMOS_HOST="${COSMOS_HOST:-127.0.0.1}"
export TEST_NUM
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

echo "[cosmos] tasks: ${TASKS}"
echo "[cosmos] episodes per task: ${TEST_NUM} | task_config=${task_config} seed=${seed} gpu=${gpu_id}"
echo "[cosmos] starting server -> ${SERVER_LOG}"
bash "${POLICY_DIR}/serve.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until grep -q "server listening on ws://.*:${PORT}/" "${SERVER_LOG}" 2>/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cosmos] server exited early; see ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[cosmos] timeout waiting for server on :${PORT}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 3
done
echo "[cosmos] server ready on :${PORT}"

# Keep going after a task fails: one broken task should not cost the whole sweep,
# and the failure is visible in the summary rather than swallowed.
failed=()
for task in ${TASKS}; do
  echo
  echo "=============================================================="
  echo "[cosmos] task=${task}"
  echo "=============================================================="
  if ! bash "${POLICY_DIR}/eval.sh" \
        "${task}" "${task_config}" "${ckpt_setting}" "${seed}" "${gpu_id}" \
        "127.0.0.1" "${PORT}"; then
    echo "[cosmos] task ${task} FAILED" >&2
    failed+=("${task}")
  fi
done

echo
echo "=============================================================="
echo "[cosmos] sweep done. results under ${EVAL_RESULT_ROOT}/<task>/cosmos/${task_config}/${EVAL_RUN_NAME}/"
if (( ${#failed[@]} > 0 )); then
  echo "[cosmos] tasks that errored: ${failed[*]}" >&2
  exit 1
fi
