#!/usr/bin/env bash
# Volcano / cloud entry for pi0.5 FM tau-schedule serial sweep.
# Sets headless Sapien render env, then runs the local serial orchestrator.
#
# Args: same as eval_pi05_tau_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# policy/pi05/eval_script -> DRRM-VODPP
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CLOUD_ENV="${ROOT_DIR}/scripts/eval/_robotwin_cloud_render_env.sh"

if [[ ! -f "${CLOUD_ENV}" ]]; then
  echo "[cloud-pi05-tau-sweep] ERROR: missing ${CLOUD_ENV}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CLOUD_ENV}"

echo -e "\033[36m[cloud-pi05-tau-sweep] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-pi05-tau-sweep] vk_icd=${VK_ICD_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-pi05-tau-sweep] egl_vendor=${__EGL_VENDOR_LIBRARY_FILENAMES:-unset}\033[0m"
echo -e "\033[36m[cloud-pi05-tau-sweep] kraken_egl_device=${KRAKEN_EGL_DEVICE:-unset}\033[0m"

exec bash "${SCRIPT_DIR}/eval_pi05_tau_sweep.sh" "$@"
