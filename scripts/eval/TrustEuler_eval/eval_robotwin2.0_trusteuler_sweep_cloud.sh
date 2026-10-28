#!/usr/bin/env bash
# Volcano / cloud entry for TrustEuler-aware RoboTwin tau sweep.
# Sets headless Sapien render env, then runs the local TrustEuler orchestrator
# with EVAL_SH pointed at the cloud parallel wrapper.
#
# Args: same as eval_robotwin2.0_trusteuler_sweep.sh
#   [--serial] [--skip-calibrate] <policy> <task> <task_config> <ckpt> ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$EVAL_DIR/_robotwin_cloud_render_env.sh"

export EVAL_SH="$EVAL_DIR/eval_robotwin2.0_parallel_cloud.sh"
export TRUSTEULER_RUNS_ROOT="${TRUSTEULER_RUNS_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/TrustEuler_runs}"

echo -e "\033[36m[cloud-trusteuler-sweep] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-trusteuler-sweep] vk_icd=${VK_ICD_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-trusteuler-sweep] egl_vendor=${__EGL_VENDOR_LIBRARY_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-trusteuler-sweep] kraken_egl_device=${KRAKEN_EGL_DEVICE:-unset}\033[0m"
echo -e "\033[36m[cloud-trusteuler-sweep] eval_sh=${EVAL_SH}\033[0m"
echo -e "\033[36m[cloud-trusteuler-sweep] trusteuler_runs=${TRUSTEULER_RUNS_ROOT}\033[0m"

exec bash "$SCRIPT_DIR/eval_robotwin2.0_trusteuler_sweep.sh" "$@"
