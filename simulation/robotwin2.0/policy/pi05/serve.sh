#!/usr/bin/env bash
# Start OpenPI pi0.5 policy server (GPU). Use from local L4 or Volcano 1-GPU jobs.
#
# Optional FM schedule (pi0.5 order 1→0):
#   PI05_FM_TAU_SCHEDULE=uniform10|uniform3|1,0.7,0.3,0
#   PI05_NUM_STEPS=10   (only if schedule unset; uniform NFE)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang/openpi}"
CKPT="${PI05_CKPT:-/DRoboticsAILab/users/licheng.yang/checkpoints/openpi/pi05_robotwin_My_clean50/robotwin_My_clean50/40000}"
CONFIG_NAME="${PI05_CONFIG:-pi05_robotwin2.0_agilex}"
PORT="${PI05_PORT:-8000}"
# Must match training multi-task repo order (asset_id / norm_stats).
export ROBOTWIN_REPO_ID="${ROBOTWIN_REPO_ID:-adjust_bottle,scan_object,beat_block_hammer}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_ROOT}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Avoid torch.compile/inductor → triton/ptxas (often broken +x on NFS mounts).
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

if [[ ! -f "${CKPT}/model.safetensors" ]]; then
  echo "ERROR: missing ${CKPT}/model.safetensors" >&2
  exit 1
fi

# Cloud RoboTwin eval may inject PYTHONPATH / ROBOTWIN_VENV_SITE (py3.10 site-packages).
# openpi server is py3.11 — that pollution makes transformers import a broken py310 sklearn.
unset ROBOTWIN_VENV_SITE
export PYTHONPATH=""

# shellcheck source=/dev/null
source "${OPENPI_ROOT}/.venv/bin/activate"
cd "${OPENPI_ROOT}"

RESOLVE_PY="${SCRIPT_DIR}/fm_schedule.py"
SERVE_EXTRA=()
if [[ -n "${PI05_FM_TAU_SCHEDULE:-}" ]]; then
  RESOLVED="$(python "${RESOLVE_PY}" "${PI05_FM_TAU_SCHEDULE}")"
  SERVE_EXTRA+=(--fm-tau-schedule="${RESOLVED}")
  echo "[pi05 serve] fm_tau_schedule=${PI05_FM_TAU_SCHEDULE} -> ${RESOLVED}"
elif [[ -n "${PI05_NUM_STEPS:-}" ]]; then
  SERVE_EXTRA+=(--num-steps="${PI05_NUM_STEPS}")
  echo "[pi05 serve] num_steps=${PI05_NUM_STEPS}"
else
  echo "[pi05 serve] FM schedule: model default (num_steps=10, uniform 1→0)"
fi

echo "[pi05 serve] config=${CONFIG_NAME}"
echo "[pi05 serve] ckpt=${CKPT}"
echo "[pi05 serve] port=${PORT} repo_id=${ROBOTWIN_REPO_ID}"

# tyro: global flags (--port, --fm-tau-schedule, --num-steps) must come before
# the policy:checkpoint subcommand.
exec python scripts/serve_policy.py \
  --port="${PORT}" \
  "${SERVE_EXTRA[@]}" \
  policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CKPT}"
