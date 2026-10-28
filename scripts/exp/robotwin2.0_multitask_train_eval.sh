#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/exp/robotwin2.0_multitask_train_eval.sh --config-dir CONFIG --policy POLICY --model MODEL [options]

Options:
  --config-dir PATH              Training config yaml
  --policy VALUE                 Eval policy name, e.g. VODPP
  --model VALUE                  Eval checkpoint setting/model directory name
  --tasks a,b,c                  Task list passed to train and eval scripts
  --demo N                       Expert/demo number used by train and eval, default: 100
  --train-num-processes N        accelerate process count for training
  --eval-num-processes N         parallel process count for eval, default: 8
  --task-configs a,b,c           Eval task configs, default: agilex_config
  --eval-checkpoint-num N        Optional eval checkpoint number
  --eval-checkpoint-root PATH    Optional eval checkpoint root override
  --skip-train                   Only run eval
  --skip-eval                    Only run train
  -h, --help                     Show this help
EOF
}

require_value() {
    local name="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "Missing value for ${name}" >&2
        usage >&2
        exit 1
    fi
}

config_dir=""
policy=""
model=""
tasks=""
demo="100"
train_num_processes=""
eval_num_processes="64"
task_configs="agilex_config"
eval_checkpoint_num=""
eval_checkpoint_root=""
skip_train=0
skip_eval=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-dir=*) config_dir="${1#*=}"; shift ;;
        --config-dir) require_value "$1" "${2:-}"; config_dir="$2"; shift 2 ;;
        --policy=*) policy="${1#*=}"; shift ;;
        --policy) require_value "$1" "${2:-}"; policy="$2"; shift 2 ;;
        --model=*) model="${1#*=}"; shift ;;
        --model) require_value "$1" "${2:-}"; model="$2"; shift 2 ;;
        --tasks=*) tasks="${1#*=}"; shift ;;
        --tasks) require_value "$1" "${2:-}"; tasks="$2"; shift 2 ;;
        --demo=*) demo="${1#*=}"; shift ;;
        --demo) require_value "$1" "${2:-}"; demo="$2"; shift 2 ;;
        --train-num-processes=*) train_num_processes="${1#*=}"; shift ;;
        --train-num-processes) require_value "$1" "${2:-}"; train_num_processes="$2"; shift 2 ;;
        --eval-num-processes=*) eval_num_processes="${1#*=}"; shift ;;
        --eval-num-processes) require_value "$1" "${2:-}"; eval_num_processes="$2"; shift 2 ;;
        --task-configs=*) task_configs="${1#*=}"; shift ;;
        --task-configs) require_value "$1" "${2:-}"; task_configs="$2"; shift 2 ;;
        --eval-checkpoint-num=*) eval_checkpoint_num="${1#*=}"; shift ;;
        --eval-checkpoint-num) require_value "$1" "${2:-}"; eval_checkpoint_num="$2"; shift 2 ;;
        --eval-checkpoint-root=*) eval_checkpoint_root="${1#*=}"; shift ;;
        --eval-checkpoint-root) require_value "$1" "${2:-}"; eval_checkpoint_root="$2"; shift 2 ;;
        --skip-train) skip_train=1; shift ;;
        --skip-eval) skip_eval=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

missing=()
if [[ ${skip_train} -eq 0 ]]; then
    [[ -n "${config_dir}" ]] || missing+=("--config-dir")
fi
if [[ ${skip_eval} -eq 0 ]]; then
    [[ -n "${policy}" ]] || missing+=("--policy")
    [[ -n "${model}" ]] || missing+=("--model")
fi
if [[ ${skip_train} -eq 1 && ${skip_eval} -eq 1 ]]; then
    echo "Nothing to do: --skip-train and --skip-eval cannot both be set" >&2
    exit 1
fi
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing required arguments: ${missing[*]}" >&2
    usage >&2
    exit 1
fi

run_or_print() {
    if [[ "${DRRM_SCRIPT_DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY_RUN wrapper command: '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ ${skip_train} -eq 0 ]]; then
    train_cmd=(
        ./scripts/train/robotwin2_benchmarking_multitask_shm.sh
        --config_dir="${config_dir}"
        --demo="${demo}"
    )
    if [[ -n "${tasks}" ]]; then
        train_cmd+=(--tasks="${tasks}")
    fi
    if [[ -n "${train_num_processes}" ]]; then
        train_cmd+=(--num_processes="${train_num_processes}")
    fi
    run_or_print "${train_cmd[@]}"
fi

if [[ ${skip_eval} -eq 0 ]]; then
    eval_cmd=(
        ./scripts/eval/eval_robotwin2.0_multitask.sh
        --policy "${policy}"
        --model "${model}"
        --train-demo "${demo}"
        --num-process "${eval_num_processes}"
        --task-configs "${task_configs}"
    )
    if [[ -n "${tasks}" ]]; then
        eval_cmd+=(--tasks "${tasks}")
    fi
    if [[ -n "${eval_checkpoint_num}" ]]; then
        eval_cmd+=(--checkpoint-num "${eval_checkpoint_num}")
    fi
    if [[ -n "${eval_checkpoint_root}" ]]; then
        eval_cmd+=(--checkpoint-root "${eval_checkpoint_root}")
    fi
    run_or_print "${eval_cmd[@]}"
fi
