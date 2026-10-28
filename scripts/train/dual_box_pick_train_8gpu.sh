#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_ACTIVATE="${ENV_ACTIVATE:-.venv/bin/activate}"

if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck source=/dev/null
    source "$ENV_ACTIVATE"
else
    echo "Environment activation script not found: ${ENV_ACTIVATE}" >&2
    echo "Set ENV_ACTIVATE=/path/to/activate if you want to use another environment." >&2
    exit 1
fi

export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Cannot find executable python at: ${PYTHON_BIN}" >&2
    echo "Current VIRTUAL_ENV: ${VIRTUAL_ENV:-}" >&2
    echo "Current PATH: ${PATH}" >&2
    exit 1
fi

echo "VIRTUAL_ENV: ${VIRTUAL_ENV}"
echo "Python: ${PYTHON_BIN}"

CONFIG_PATH="${CONFIG_PATH:-configs/vodpp_train/dual_box_pick_8gpu.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CMD=(
    "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${NUM_PROCESSES}"
    --gpu_ids "${GPU_IDS}"
    main.py
    "--config-path=${CONFIG_PATH%/*}"
    "--config-name=$(basename "$CONFIG_PATH")"
)

echo "Starting dual_box_pick 8-GPU training"
echo "Config: ${CONFIG_PATH}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "GPUs: ${GPU_IDS} (${NUM_PROCESSES} processes)"
printf 'Command:'
printf ' %q' "${CMD[@]}" "$@"
printf '\n'

"${CMD[@]}" "$@"
