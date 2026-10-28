#!/usr/bin/env bash
# Volcano / cloud entry for the pi0.5 N-GPU closed-loop eval.
# Sets the headless Sapien render env, then runs the multi-GPU orchestrator.
# Args: same as eval_pi05_multi_gpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CLOUD_ENV="${ROOT_DIR}/scripts/eval/_robotwin_cloud_render_env.sh"

if [[ ! -f "${CLOUD_ENV}" ]]; then
  echo "[cloud-pi05-mgpu] ERROR: missing ${CLOUD_ENV}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CLOUD_ENV}"

echo -e "\033[36m[cloud-pi05-mgpu] python=${PYTHON:-unset}\033[0m"
echo -e "\033[36m[cloud-pi05-mgpu] gpus=${PI05_GPUS:-4} vk_icd=${VK_ICD_FILENAMES:-unset}\033[0m"
nvidia-smi -L || true

exec bash "${SCRIPT_DIR}/eval_pi05_multi_gpu.sh" "$@"
