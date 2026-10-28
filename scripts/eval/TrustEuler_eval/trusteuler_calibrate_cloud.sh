#!/usr/bin/env bash
# Volcano / cloud entry: TrustEuler U100 dense-trace collection only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$EVAL_DIR/_robotwin_cloud_render_env.sh"

export TRUSTEULER_RUNS_ROOT="${TRUSTEULER_RUNS_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs}"
export DEVICE="${DEVICE:-cuda:${GPU_ID:-0}}"

echo -e "\033[36m[cloud-trusteuler-calibrate] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-trusteuler-calibrate] device=${DEVICE}\033[0m"
echo -e "\033[36m[cloud-trusteuler-calibrate] trusteuler_runs=${TRUSTEULER_RUNS_ROOT}\033[0m"

exec bash "$SCRIPT_DIR/trusteuler_calibrate.sh" "$@"
