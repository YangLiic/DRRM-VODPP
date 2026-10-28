#!/usr/bin/env bash
# Volcano / cloud entry for a single RoboTwin eval.
# Sets headless Sapien render env, then runs the local parallel evaluator.
#
# Args: same as eval_robotwin2.0_parallel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_robotwin_cloud_render_env.sh"

echo -e "\033[36m[cloud-render] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-render] vk_icd=${VK_ICD_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-render] egl_vendor=${__EGL_VENDOR_LIBRARY_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-render] kraken_egl_device=${KRAKEN_EGL_DEVICE:-unset}\033[0m"

exec bash "$SCRIPT_DIR/eval_robotwin2.0_parallel.sh" "$@"
