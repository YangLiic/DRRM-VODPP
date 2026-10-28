#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-configs/vodpp_train/one_box_pick_rgb_8gpu.yaml}"

exec "${SCRIPT_DIR}/one_box_pick_train_8gpu.sh" "$@"
