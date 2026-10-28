#!/usr/bin/env bash
# Universal training launcher. Usage:
#   ./scripts/train/thrain.sh [config_path] [extra hydra overrides...]
#
# config_path: path to the yaml (default: configs/vodpp_train/three_boxL.yaml)
#   Can also be set via CONFIG_PATH env var; positional arg takes precedence.
#
# Examples:
#   ./scripts/train/thrain.sh configs/vodpp_train/three_boxL.yaml
#   CONFIG_PATH=configs/vodpp_train/two_box_pick_video_8gpu.yaml ./scripts/train/thrain.sh
#   TRAIN_CACHE_DECODED_VIDEO=true ./scripts/train/thrain.sh configs/vodpp_train/three_boxL.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
source "${SCRIPT_DIR}/episode_cache_server_utils.sh"

export VIRTUAL_ENV="/data/vepfs/licheng.yang/DRRM-VODPP/.venv"
export PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Cannot find executable python at: ${PYTHON_BIN}" >&2
    exit 1
fi

# ── Config: positional arg > env var > default ──────────────────────────────
if [ $# -ge 1 ] && [[ "$1" != --* ]] && [[ "$1" != ++* ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="${CONFIG_PATH:-configs/vodpp_train/three_boxL.yaml}"
fi

# Derive a short tag from the yaml stem for log dirs and socket names.
config_stem="${CONFIG_PATH##*/}"   # e.g. three_boxL.yaml
config_stem="${config_stem%.yaml}" # e.g. three_boxL

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-lazy}"
TRAIN_CACHE_MAX_EPISODES="${TRAIN_CACHE_MAX_EPISODES:-}"
TRAIN_CACHE_REPORT_INTERVAL="${TRAIN_CACHE_REPORT_INTERVAL:-1000}"
TRAIN_CACHE_BUILD_NUM_WORKERS="${TRAIN_CACHE_BUILD_NUM_WORKERS:-8}"
TRAIN_CACHE_SOCKET="${TRAIN_CACHE_SOCKET:-}"
TRAIN_CACHE_DECODED_VIDEO="${TRAIN_CACHE_DECODED_VIDEO:-true}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

# ── Cache-server dirs/sockets derived from config stem (no manual edits) ────
CACHE_LOG_DIR="${CACHE_LOG_DIR:-/tmp/drrm_${config_stem}_cache}"
mkdir -p "${CACHE_LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
socket_token="${SLURM_JOB_ID:-$$}_${timestamp}"
if [ -n "$TRAIN_CACHE_SOCKET" ]; then
    current_cache_socket="$TRAIN_CACHE_SOCKET"
else
    current_cache_socket="/tmp/drrm_${socket_token}_${config_stem}.sock"
fi

cache_server_log_file="${CACHE_LOG_DIR}/episode_cache_server.log"
cache_server_ready_file="${CACHE_LOG_DIR}/episode_cache_server.ready"
cache_server_cmd_file="${CACHE_LOG_DIR}/episode_cache_server_command.txt"
CACHE_SERVER_PID=""

cleanup_cache_server() {
    stop_shm_episode_cache_server "${current_cache_socket:-}" "${CACHE_SERVER_PID:-}"
}
trap cleanup_cache_server EXIT

rm -f "$cache_server_ready_file"

config_name="$(basename "$CONFIG_PATH")"
start_shm_episode_cache_server \
    "$CONFIG_PATH" \
    "$config_name" \
    "$current_cache_socket" \
    "$cache_server_log_file" \
    "$cache_server_ready_file" \
    "$cache_server_cmd_file" \
    "$TRAIN_CACHE_REPORT_INTERVAL" \
    "$TRAIN_CACHE_BUILD_NUM_WORKERS" \
    "$TRAIN_CACHE_MAX_EPISODES" \
    "" \
    "" \
    "++train_dataset.cache_decoded_video=${TRAIN_CACHE_DECODED_VIDEO}"
CACHE_SERVER_PID="$EPISODE_CACHE_SERVER_PID"

hydra_overrides=(
    "++train_dataset.cache_to_ram=true"
    "++train_dataset.cache_mode=${TRAIN_CACHE_MODE}"
    "++train_dataset.cache_backend=shm"
    "++train_dataset.cache_socket=${current_cache_socket}"
    "++train_dataset.cache_report_interval=${TRAIN_CACHE_REPORT_INTERVAL}"
    "++train_dataset.cache_decoded_video=${TRAIN_CACHE_DECODED_VIDEO}"
)
if [ -n "$TRAIN_CACHE_MAX_EPISODES" ]; then
    hydra_overrides+=("++train_dataset.cache_max_episodes=${TRAIN_CACHE_MAX_EPISODES}")
fi

CMD=(
    "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    --gpu_ids "${GPU_IDS}"
    main.py
    "--config-path=${CONFIG_PATH%/*}"
    "--config-name=${config_name}"
    "${hydra_overrides[@]}"
)

echo "Starting ${config_stem} training (shm cache)"
echo "Config: ${CONFIG_PATH} | GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_PROCESSES} processes)"
echo "output_dir: from yaml (base_dir)"
echo "cache_socket=${current_cache_socket}, cache_mode=${TRAIN_CACHE_MODE}, build_workers=${TRAIN_CACHE_BUILD_NUM_WORKERS}"
printf 'Command:'
printf ' %q' "${CMD[@]}" "$@"
printf '\n'

"${CMD[@]}" "$@"
status=$?

stop_shm_episode_cache_server "$current_cache_socket" "$CACHE_SERVER_PID"
CACHE_SERVER_PID=""
exit "$status"
