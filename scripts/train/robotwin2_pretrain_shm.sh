#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/episode_cache_server_utils.sh"

CONFIG_PATH="${1:?Usage: $0 <config.yaml>}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-lazy}"
TRAIN_CACHE_SOCKET="${TRAIN_CACHE_SOCKET:-}"
TRAIN_CACHE_MAX_EPISODES="${TRAIN_CACHE_MAX_EPISODES:-}"
TRAIN_CACHE_REPORT_INTERVAL="${TRAIN_CACHE_REPORT_INTERVAL:-1000}"
TRAIN_CACHE_BUILD_NUM_WORKERS="${TRAIN_CACHE_BUILD_NUM_WORKERS:-1}"

CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.*}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SOCKET_TOKEN="${SLURM_JOB_ID:-$$}_${TIMESTAMP}"
RUN_DIR="${TRAIN_CACHE_RUN_DIR:-outputs/train_shm_${CONFIG_STEM}_${TIMESTAMP}}"
LOG_FILE="${RUN_DIR}/train.log"
ENV_FILE="${RUN_DIR}/environment.txt"
CMD_FILE="${RUN_DIR}/command.txt"
CACHE_SERVER_LOG_FILE="${RUN_DIR}/episode_cache_server.log"
CACHE_SERVER_READY_FILE="${RUN_DIR}/episode_cache_server.ready"
CACHE_SERVER_CMD_FILE="${RUN_DIR}/episode_cache_server_command.txt"
if [ -z "$TRAIN_CACHE_SOCKET" ]; then
    TRAIN_CACHE_SOCKET="/tmp/drrm_${SOCKET_TOKEN}.sock"
fi

mkdir -p "$RUN_DIR"

{
    echo "timestamp: ${TIMESTAMP}"
    echo "host: $(hostname)"
    echo "pwd: $(pwd)"
    echo "config: ${CONFIG_PATH}"
    echo "cache_socket: ${TRAIN_CACHE_SOCKET}"
    echo
    echo "git:"
    git rev-parse --short HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "python:"
    python -V 2>&1 || true
    python - <<'PY' 2>/dev/null || true
import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(idx)
    print(f"gpu[{idx}]: {prop.name}, total_memory={prop.total_memory / 1024**3:.1f}GiB")
PY
    echo
    echo "accelerate:"
    accelerate env 2>/dev/null || true
} > "$ENV_FILE"

HYDRA_OVERRIDES=(
    "++train_dataset.cache_to_ram=true"
    "++train_dataset.cache_mode=${TRAIN_CACHE_MODE}"
    "++train_dataset.cache_backend=shm"
    "++train_dataset.cache_socket=${TRAIN_CACHE_SOCKET}"
    "++train_dataset.cache_report_interval=${TRAIN_CACHE_REPORT_INTERVAL}"
)
if [ -n "$TRAIN_CACHE_MAX_EPISODES" ]; then
    HYDRA_OVERRIDES+=("++train_dataset.cache_max_episodes=${TRAIN_CACHE_MAX_EPISODES}")
fi
CMD=(
    accelerate launch
    --config_file configs/accelerate_config.yaml
    main.py
    "--config-path=${CONFIG_PATH%/*}"
    "--config-name=${CONFIG_NAME}"
    "${HYDRA_OVERRIDES[@]}"
)

printf '%q ' "${CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

echo "Train output: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Episode cache server log: ${CACHE_SERVER_LOG_FILE}"
echo "Environment: ${ENV_FILE}"
echo "Command: ${CMD_FILE}"

CACHE_SERVER_PID=""
cleanup_cache_server() {
    stop_shm_episode_cache_server "$TRAIN_CACHE_SOCKET" "$CACHE_SERVER_PID"
}
trap cleanup_cache_server EXIT

start_shm_episode_cache_server \
    "$CONFIG_PATH" \
    "$CONFIG_NAME" \
    "$TRAIN_CACHE_SOCKET" \
    "$CACHE_SERVER_LOG_FILE" \
    "$CACHE_SERVER_READY_FILE" \
    "$CACHE_SERVER_CMD_FILE" \
    "$TRAIN_CACHE_REPORT_INTERVAL" \
    "$TRAIN_CACHE_BUILD_NUM_WORKERS" \
    "$TRAIN_CACHE_MAX_EPISODES" \
    "" \
    ""
CACHE_SERVER_PID="$EPISODE_CACHE_SERVER_PID"

SECONDS=0
set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

{
    echo "exit_status: ${STATUS}"
    echo "elapsed_seconds: ${SECONDS}"
    echo "run_dir: ${RUN_DIR}"
} | tee "${RUN_DIR}/summary.txt"

exit "$STATUS"
