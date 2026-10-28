#!/usr/bin/env bash
# Client-only RoboTwin eval; expects the Cosmos policy server to be up already.
# Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> [host] [port]
set -euo pipefail

policy_name=cosmos
task_name=${1:?task_name}
task_config=${2:?task_config}
ckpt_setting=${3:?ckpt_setting}
seed=${4:-0}
gpu_id=${5:-0}
host=${6:-${COSMOS_HOST:-127.0.0.1}}
port=${7:-${COSMOS_PORT:-8000}}
replan_steps=${COSMOS_REPLAN_STEPS:-auto}
prompt_mode=${COSMOS_PROMPT_MODE:-env}
instruction_type=${INSTRUCTION_TYPE:-seen}
test_num=${TEST_NUM:-100}
run_name=${EVAL_RUN_NAME:-${ckpt_setting}}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m[cosmos eval] gpu=${gpu_id} server=${host}:${port} replan=${replan_steps} prompt=${prompt_mode} run=${run_name}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${ROBOTWIN_ROOT}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SAPIEN_EGL_NO_X11="${SAPIEN_EGL_NO_X11:-1}"

# Same local Sapien Vulkan ICD fix the pi0.5 arm uses: without it Sapien picks a
# software ICD (or none) and rendering either crashes or silently goes blank.
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  mkdir -p /tmp/vulkan_icd
  _vk_lib=""
  for _cand in \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so \
    $(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* 2>/dev/null | grep -v '\.so\.0$' | tail -n 1)
  do
    if [[ -n "${_cand}" && -e "${_cand}" ]]; then
      _vk_lib="${_cand}"
      break
    fi
  done
  if [[ -n "${_vk_lib}" ]]; then
    cat > /tmp/vulkan_icd/nvidia_egl.json <<VKEOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${_vk_lib}",
        "api_version" : "1.3.242"
    }
}
VKEOF
    export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
  elif [[ -f /tmp/vulkan_icd/nvidia_egl.json ]]; then
    export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
  fi
  unset _vk_lib _cand
fi
echo -e "\033[36m[cosmos eval] VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}\033[0m"

RESULT_ROOT="${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
echo -e "\033[36m[cosmos eval] result_root=${RESULT_ROOT}\033[0m"

if [[ -z "${PYTHON:-}" && -x "${ROBOTWIN_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROBOTWIN_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
echo -e "\033[36m[cosmos eval] python=${PYTHON_BIN}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON_BIN}" script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
  --overrides \
  --task_name "${task_name}" \
  --task_config "${task_config}" \
  --ckpt_setting "${ckpt_setting}" \
  --seed "${seed}" \
  --policy_name "${policy_name}" \
  --host "${host}" \
  --port "${port}" \
  --replan_steps "${replan_steps}" \
  --prompt_mode "${prompt_mode}" \
  --instruction_type "${instruction_type}" \
  --test_num "${test_num}" \
  --eval_result_root "${RESULT_ROOT}" \
  --eval_run_name "${run_name}"
