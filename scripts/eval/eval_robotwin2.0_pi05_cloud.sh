#!/usr/bin/env bash
# Cloud / Volcano entry for pi0.5 RoboTwin eval (1x GPU: server + client).
# Usage (from DRRM-VODPP root, as job command):
#   bash scripts/eval/eval_robotwin2.0_pi05_cloud.sh adjust_bottle
#   bash scripts/eval/eval_robotwin2.0_pi05_cloud.sh scan_object agilex_config 3task_25000 0 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
if [[ -f "${SCRIPT_DIR}/_robotwin_cloud_render_env.sh" ]]; then
  source "${SCRIPT_DIR}/_robotwin_cloud_render_env.sh"
fi

cd "${ROOT_DIR}"
exec bash simulation/robotwin2.0/policy/pi05/run_eval.sh "$@"
