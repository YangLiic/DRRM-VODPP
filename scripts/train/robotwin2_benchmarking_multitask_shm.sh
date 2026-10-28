#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/episode_cache_server_utils.sh"

split_csv_into_array() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "${value}"
}

demo=100
config_dir=""
task=""
num_processes=""
positional_args=()
task_list=()
custom_tasks=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir=*) config_dir="${1#*=}"; shift ;;
        --task=*) task_list+=("${1#*=}"); custom_tasks=1; shift ;;
        --tasks=*) split_csv_into_array "${1#*=}" task_list; custom_tasks=1; shift ;;
        --num_processes=*) num_processes="${1#*=}"; shift ;;
        --demo=*) demo="${1#*=}"; shift ;;
        --*) echo "Unknown parameter: $1"; exit 1 ;;
        *) positional_args+=("$1"); shift ;;
    esac
done

if [ -z "$config_dir" ] && [ ${#positional_args[@]} -ge 1 ]; then config_dir="${positional_args[0]}"; fi
if [ ${custom_tasks} -eq 0 ] && [ ${#positional_args[@]} -ge 2 ]; then task="${positional_args[1]}"; fi
if [ -z "$num_processes" ] && [ ${#positional_args[@]} -ge 3 ]; then num_processes="${positional_args[2]}"; fi
if [ ${#positional_args[@]} -ge 4 ]; then demo="${positional_args[3]}"; fi

if [ -z "$config_dir" ]; then
    echo "Usage: $0 --config_dir=<config.yaml> [--task=<task>|--tasks=a,b,c] [--num_processes=N] [--demo=N]" >&2
    echo "   or: $0 <config.yaml> [task] [num_processes] [demo]" >&2
    exit 1
fi
if [ ! -f "$config_dir" ]; then
    echo "Config not found: $config_dir" >&2
    exit 1
fi

if [ ${custom_tasks} -eq 1 ]; then
    if [ ${#task_list[@]} -eq 0 ]; then
        echo "No tasks specified" >&2
        exit 1
    fi
elif [ -n "$task" ]; then
    task_list=("$task")
else
    task_list=(
        # 'open_laptop' 'move_stapler_pad' 'press_stapler'
        # 'place_a2b_left' 'stack_blocks_two'
        'hanging_mug' 'beat_block_hammer' 'dump_bin_bigbin'
        'handover_block' 'blocks_ranking_rgb'
    )
fi

num_args=()
if [ -n "$num_processes" ]; then
    num_args=("--num_processes=$num_processes")
fi

TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-lazy}"
TRAIN_CACHE_MAX_EPISODES="${TRAIN_CACHE_MAX_EPISODES:-}"
TRAIN_CACHE_REPORT_INTERVAL="${TRAIN_CACHE_REPORT_INTERVAL:-1000}"
TRAIN_CACHE_BUILD_NUM_WORKERS="${TRAIN_CACHE_BUILD_NUM_WORKERS:-8}"
TRAIN_CACHE_SOCKET="${TRAIN_CACHE_SOCKET:-}"

config_name="${config_dir##*/}"
config="${config_name%.*}"
timestamp="$(date +%Y%m%d_%H%M%S)"
socket_token="${SLURM_JOB_ID:-$$}_${timestamp}"
base_run_dir="${TRAIN_CACHE_RUN_DIR:-outputs/benchmark_multitask_shm_${config}_${timestamp}}"
mkdir -p "$base_run_dir"

cleanup_cache_server() {
    stop_shm_episode_cache_server "${current_cache_socket:-}" "${CACHE_SERVER_PID:-}"
}
trap cleanup_cache_server EXIT

for task in "${task_list[@]}"; do
    echo "Training task: $task ($demo) with config: $config_dir"

    task_run_dir="${base_run_dir}/${task}_${demo}"
    mkdir -p "$task_run_dir"

    log_file="${task_run_dir}/train.log"
    env_file="${task_run_dir}/environment.txt"
    cmd_file="${task_run_dir}/command.txt"
    cache_server_log_file="${task_run_dir}/episode_cache_server.log"
    cache_server_ready_file="${task_run_dir}/episode_cache_server.ready"
    cache_server_cmd_file="${task_run_dir}/episode_cache_server_command.txt"
    if [ -n "$TRAIN_CACHE_SOCKET" ]; then
        current_cache_socket="$TRAIN_CACHE_SOCKET"
    else
        current_cache_socket="/tmp/drrm_${socket_token}_${task}.sock"
    fi

    if [ "${DRRM_SCRIPT_DRY_RUN:-0}" = "1" ]; then
        hydra_overrides=(
            "train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb"
            "train_dataset.task=${task}"
            "train_dataset.demo=${demo}"
            "++train_dataset.cache_to_ram=true"
            "++train_dataset.cache_mode=${TRAIN_CACHE_MODE}"
            "++train_dataset.cache_backend=shm"
            "++train_dataset.cache_socket=${current_cache_socket}"
            "++train_dataset.cache_report_interval=${TRAIN_CACHE_REPORT_INTERVAL}"
        )
        if [ -n "$TRAIN_CACHE_MAX_EPISODES" ]; then
            hydra_overrides+=("++train_dataset.cache_max_episodes=${TRAIN_CACHE_MAX_EPISODES}")
        fi
        cmd=(
            accelerate launch
            --config_file configs/accelerate_config.yaml
            "${num_args[@]}"
            main.py
            "--config-path=${config_dir%/*}"
            "--config-name=${config_dir##*/}"
            "${hydra_overrides[@]}"
        )
        echo "DRY_RUN train task: ${task}"
        printf 'DRY_RUN command: '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    {
        echo "timestamp: ${timestamp}"
        echo "host: $(hostname)"
        echo "pwd: $(pwd)"
        echo "config: ${config_dir}"
        echo "task: ${task}"
        echo "demo: ${demo}"
        echo "cache_socket: ${current_cache_socket}"
        echo
        echo "git:"
        git rev-parse --short HEAD 2>/dev/null || true
        git status --short 2>/dev/null || true
        echo
        echo "python:"
        python -V 2>&1 || true
        python - <<'PY' 2>/dev/null || true
import torch
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(idx)
    print(f"gpu[{idx}]: {prop.name}, total_memory={prop.total_memory / 1024**3:.1f}GiB")
PY
        echo
        echo "accelerate:"
        accelerate env 2>/dev/null || true
    } > "$env_file"

    hydra_overrides=(
        "train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb"
        "train_dataset.task=${task}"
        "train_dataset.demo=${demo}"
        "++train_dataset.cache_to_ram=true"
        "++train_dataset.cache_mode=${TRAIN_CACHE_MODE}"
        "++train_dataset.cache_backend=shm"
        "++train_dataset.cache_socket=${current_cache_socket}"
        "++train_dataset.cache_report_interval=${TRAIN_CACHE_REPORT_INTERVAL}"
    )
    if [ -n "$TRAIN_CACHE_MAX_EPISODES" ]; then
        hydra_overrides+=("++train_dataset.cache_max_episodes=${TRAIN_CACHE_MAX_EPISODES}")
    fi

    cmd=(
        accelerate launch
        --config_file configs/accelerate_config.yaml
        "${num_args[@]}"
        main.py
        "--config-path=${config_dir%/*}"
        "--config-name=${config_dir##*/}"
        "${hydra_overrides[@]}"
    )

    printf '%q ' "${cmd[@]}" > "$cmd_file"
    printf '\n' >> "$cmd_file"

    if [ "${DRRM_SCRIPT_DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN train task: ${task}"
        printf 'DRY_RUN command: '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        continue
    fi

    CACHE_SERVER_PID=""
    start_shm_episode_cache_server \
        "$config_dir" \
        "$config_name" \
        "$current_cache_socket" \
        "$cache_server_log_file" \
        "$cache_server_ready_file" \
        "$cache_server_cmd_file" \
        "$TRAIN_CACHE_REPORT_INTERVAL" \
        "$TRAIN_CACHE_BUILD_NUM_WORKERS" \
        "$TRAIN_CACHE_MAX_EPISODES" \
        "$task" \
        "$demo" \
        "train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb"
    CACHE_SERVER_PID="$EPISODE_CACHE_SERVER_PID"

    SECONDS=0
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
    set -e

    {
        echo "exit_status: ${status}"
        echo "elapsed_seconds: ${SECONDS}"
        echo "run_dir: ${task_run_dir}"
    } | tee "${task_run_dir}/summary.txt"

    stop_shm_episode_cache_server "$current_cache_socket" "$CACHE_SERVER_PID"
    CACHE_SERVER_PID=""

    if [ "$status" -ne 0 ]; then
        exit "$status"
    fi
done
