#!/usr/bin/env bash
# Start GR00T N1.7 policy server (GPU). Use from local L4 or Volcano 1-GPU jobs.
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/DRoboticsAILab/vepfs_20260622/licheng.yang/Isaac-GR00T}"
CKPT="${GR00T_CKPT:-/DRoboticsAILab/users/licheng.yang/checkpoints/gr00t/n17_robotwin50_clean_bgr2rgb_h32/n17_robotwin50_clean_bgr2rgb_h32/checkpoint-40000}"
EMBODIMENT="${GR00T_EMBODIMENT:-NEW_EMBODIMENT}"
HOST="${GR00T_HOST:-127.0.0.1}"
PORT="${GR00T_PORT:-5555}"
DEVICE="${GR00T_DEVICE:-cuda}"
COSMOS="${GR00T_COSMOS_MODEL_NAME:-${GR00T_ROOT}/checkpoints/nvidia/Cosmos-Reason2-2B}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${GR00T_ROOT}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export GR00T_COSMOS_MODEL_NAME="${COSMOS}"

if [[ ! -f "${CKPT}/model.safetensors.index.json" && ! -f "${CKPT}/model.safetensors" ]]; then
  echo "ERROR: missing weights in ${CKPT}" >&2
  exit 1
fi
# The tree's own .venv symlinks its interpreter to /root/miniconda3, which is
# container-local: it exists on the dev box and on none of the 8-GPU workers, while the
# 15G of site-packages already sits on EFS.  GR00T_PY lets a caller point at an
# interpreter that every box can see; unset, this is exactly the old behaviour.
GR00T_PY="${GR00T_PY:-${GR00T_ROOT}/.venv/bin/python}"
if [[ ! -x "${GR00T_PY}" ]]; then
  echo "ERROR: missing interpreter ${GR00T_PY} (set GR00T_PY to one on shared storage)" >&2
  exit 1
fi

# Cloud RoboTwin eval may inject PYTHONPATH / ROBOTWIN_VENV_SITE (py3.10).
# GR00T server is py3.12 — drop that pollution.
unset ROBOTWIN_VENV_SITE PYTHONHOME
# When GR00T_PY is an interpreter outside the venv it cannot see the venv's packages,
# so they go on PYTHONPATH explicitly.  Harmless in the default case.
export PYTHONPATH="${GR00T_ROOT}/.venv/lib/python3.12/site-packages:${GR00T_ROOT}"

if [[ -x "${GR00T_ROOT}/examples/RoboTwin/scripts/link_local_hf_checkpoints.sh" ]]; then
  bash "${GR00T_ROOT}/examples/RoboTwin/scripts/link_local_hf_checkpoints.sh" "${GR00T_ROOT}" \
    || echo "[gr00t serve] warn: cosmos snapshot link failed (offline load may still work)" >&2
fi

cd "${GR00T_ROOT}"
echo "[gr00t serve] ckpt=${CKPT}"
echo "[gr00t serve] embodiment=${EMBODIMENT} device=${DEVICE} ${HOST}:${PORT}"
echo "[gr00t serve] cosmos=${GR00T_COSMOS_MODEL_NAME}"
echo "[gr00t serve] ROBOTWIN_BGR2RGB=${ROBOTWIN_BGR2RGB:-0}"

exec "${GR00T_PY}" -m gr00t.eval.run_gr00t_server \
  --model-path="${CKPT}" \
  --embodiment-tag="${EMBODIMENT}" \
  --device="${DEVICE}" \
  --host="${HOST}" \
  --port="${PORT}"
