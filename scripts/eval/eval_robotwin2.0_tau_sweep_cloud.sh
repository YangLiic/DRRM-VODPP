#!/usr/bin/env bash
# Volcano / cloud entry for FM tau-schedule sweep.
# Sets headless Sapien render env, then runs the local tau_sweep orchestrator
# with EVAL_SH pointed at the cloud parallel wrapper.
#
# Args: same as eval_robotwin2.0_tau_sweep.sh (including optional --serial)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_robotwin_cloud_render_env.sh"

export EVAL_SH="$SCRIPT_DIR/eval_robotwin2.0_parallel_cloud.sh"

echo -e "\033[36m[cloud-tau-sweep] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-tau-sweep] vk_icd=${VK_ICD_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-tau-sweep] egl_vendor=${__EGL_VENDOR_LIBRARY_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-tau-sweep] kraken_egl_device=${KRAKEN_EGL_DEVICE:-unset}\033[0m"
echo -e "\033[36m[cloud-tau-sweep] eval_sh=${EVAL_SH}\033[0m"

exec bash "$SCRIPT_DIR/eval_robotwin2.0_tau_sweep.sh" "$@"
