#!/usr/bin/env bash
# Volcano / cloud entry for mine_eval_seeds.py.
# Sets the headless Sapien render env (same as every *_cloud.sh), then runs the
# expert-only seed miner. Args pass through unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# script/ -> robotwin2.0 -> simulation -> DRRM-VODPP
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLOUD_ENV="${ROOT_DIR}/scripts/eval/_robotwin_cloud_render_env.sh"

if [[ ! -f "${CLOUD_ENV}" ]]; then
  echo "[mine-seeds-cloud] ERROR: missing ${CLOUD_ENV}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CLOUD_ENV}"

echo "[mine-seeds-cloud] python=${PYTHON:-unset}"
cd "${SCRIPT_DIR}/.."
exec "${PYTHON}" script/mine_eval_seeds.py "$@"
