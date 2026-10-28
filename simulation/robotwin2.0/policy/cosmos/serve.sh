#!/usr/bin/env bash
# Thin wrapper: the server lives with the model code, not with the simulator.
set -euo pipefail
COSMOS_RW_PROJECT="${COSMOS_RW_PROJECT:-/DRoboticsAILab/vepfs_20260622/licheng.yang/cosmos-robotwin}"
exec bash "${COSMOS_RW_PROJECT}/scripts/serve.sh" "$@"
