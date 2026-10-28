#!/bin/bash

# == Required arguments ==
policy_name=${1}
task_name=${2}
task_config=${3}
checkpoint=${4}
expert_data_num=${5}
num_process=${6}
seed=${7}

# == Optional arguments ==
# mode: "render" for real-time GUI rendering, "headless" (default) for no GUI
mode=${8:-headless}
# save_video: "true" (default) to save mp4 recordings, "false" to skip
save_video=${9:-true}
# test_num: number of evaluation episodes (default: 100)
test_num=${10:-100}
# fm_tau_schedule: Euler tau knots. Examples:
#   uniform10          -> same as default 10-step uniform ODE
#   0,0.25,0.5,1       -> 3 NFE (VODPP FM is 0->1, no 1-tau)
#   empty / unset      -> use checkpoint num_inference_timesteps (uniform)
fm_tau_schedule=${11:-}

DEBUG=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROBOTWIN_DIR="$PROJECT_ROOT/simulation/robotwin2.0"
cd "$ROBOTWIN_DIR"

# Prefer an explicit PYTHON (cloud wrapper sets this), then robotwin venv / conda / PATH.
# Skip interpreters that cannot import torch — unless PYTHON was explicitly exported
# (Volcano: host-bound .venv symlink is broken; wrapper points at image python3.10).
_pick_python() {
    local candidate
    if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
        echo "${PYTHON}"
        return 0
    fi
    for candidate in \
        "$ROBOTWIN_DIR/.venv/bin/python" \
        "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
        "$(command -v python3 || true)" \
        "$(command -v python || true)"
    do
        if [ -n "$candidate" ] && [ -x "$candidate" ] \
            && "$candidate" -c "import torch" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if _picked="$(_pick_python)"; then
    PYTHON="$_picked"
else
    PYTHON="$(command -v python3 || command -v python)"
    echo -e "\033[33m[Warning] No python with torch found; falling back to ${PYTHON}\033[0m"
fi
unset _picked
echo -e "\033[36m[Python] ${PYTHON}\033[0m"

export PYTHONPATH="${PROJECT_ROOT}:${ROBOTWIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Local Sapien on this fleet needs absolute libEGL ICD (stock /etc libGLX ICD fails).
# Cloud-only knobs (KRAKEN / GLVND) stay in eval_robotwin2.0_*_cloud.sh.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    mkdir -p /tmp/vulkan_icd
    _vk_lib=""
    for _cand in \
        /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
        /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so \
        $(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* 2>/dev/null | grep -v '\.so\.0$' | tail -n 1)
    do
        if [ -n "${_cand}" ] && [ -e "${_cand}" ]; then
            _vk_lib="${_cand}"
            break
        fi
    done
    if [ -n "${_vk_lib}" ]; then
        cat > /tmp/vulkan_icd/nvidia_egl.json <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${_vk_lib}",
        "api_version" : "1.3.242"
    }
}
EOF
        export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
    elif [ -f /tmp/vulkan_icd/nvidia_egl.json ]; then
        export VK_ICD_FILENAMES=/tmp/vulkan_icd/nvidia_egl.json
    fi
    unset _vk_lib _cand
fi

# Allow absolute or relative checkpoint paths.
checkpoint_path="${checkpoint}"
if [ -n "${checkpoint}" ]; then
    if [[ "${checkpoint}" = /* ]]; then
        checkpoint_path="${checkpoint}"
    elif [ -e "$PROJECT_ROOT/${checkpoint}" ]; then
        checkpoint_path="$(cd "$(dirname "$PROJECT_ROOT/${checkpoint}")" && pwd)/$(basename "${checkpoint}")"
    elif [ -e "$PROJECT_ROOT/simulation/robotwin2.0/${checkpoint}" ]; then
        checkpoint_path="$(cd "$(dirname "$PROJECT_ROOT/simulation/robotwin2.0/${checkpoint}")" && pwd)/$(basename "${checkpoint}")"
    else
        checkpoint_path="${checkpoint}"
        echo -e "\033[33m[Warning] Checkpoint path does not exist yet: ${checkpoint_path}\033[0m"
    fi
fi

args=(
    "script/eval_policy_parallel.py"
    "--config" "policy/$policy_name/deploy_policy.yml"
    "--overrides"
    "--task_name" "${task_name}"
    "--task_config" "${task_config}"
    "--ckpt_setting" "${checkpoint_path}"
    "--expert_data_num" "${expert_data_num}"
)

if [ -n "${num_process}" ]; then
    args+=("--num_process" "${num_process}")
fi

if [ -n "${seed}" ]; then
    args+=("--seed" "${seed}")
fi

if [ "${mode}" = "render" ]; then
    args+=("--render_freq" "1")
    echo -e "\033[36m[Mode] Real-time rendering enabled (render_freq=1)\033[0m"
else
    args+=("--render_freq" "0")
    echo -e "\033[36m[Mode] Headless mode (render_freq=0)\033[0m"
fi

if [ "${save_video}" = "false" ]; then
    args+=("--eval_video_log" "False")
    echo -e "\033[36m[Video] MP4 recording disabled\033[0m"
else
    args+=("--eval_video_log" "True")
    echo -e "\033[36m[Video] MP4 recording enabled\033[0m"
fi

args+=("--test_num" "${test_num}")
if [ -n "${fm_tau_schedule}" ]; then
    args+=("--fm_tau_schedule" "${fm_tau_schedule}")
    echo -e "\033[36m[FM tau] ${fm_tau_schedule}\033[0m"
fi
# Optional result relocation (used by tau8 multi-GPU launcher)
if [ -n "${EVAL_RESULT_ROOT:-}" ]; then
    args+=("--eval_result_root" "${EVAL_RESULT_ROOT}")
    echo -e "\033[36m[Result root] ${EVAL_RESULT_ROOT}\033[0m"
fi
if [ -n "${EVAL_RUN_NAME:-}" ]; then
    args+=("--eval_run_name" "${EVAL_RUN_NAME}")
    echo -e "\033[36m[Run name] ${EVAL_RUN_NAME}\033[0m"
fi
echo -e "\033[36m[Checkpoint] ${checkpoint_path}\033[0m"
echo -e "\033[36m[Episodes] ${test_num} evaluation episodes\033[0m"
echo ""

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON" "${args[@]}"
