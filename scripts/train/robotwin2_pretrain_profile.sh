#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/episode_cache_server_utils.sh"

POLICY="${POLICY:-vodpp_train}"
MODEL_CONFIG="${MODEL_CONFIG:-dit_fm_wam_dinov3cnnl_1v_10tasks_pt}"
CONFIG_PATH="${1:-configs/${POLICY}/${MODEL_CONFIG}.yaml}"
shift || true

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

PROFILE_EPOCHS="${PROFILE_EPOCHS:-1}"
PROFILE_DEMO="${PROFILE_DEMO:-}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-}"
PROFILE_VAL_BATCH_SIZE="${PROFILE_VAL_BATCH_SIZE:-}"
PROFILE_WORKERS="${PROFILE_WORKERS:-}"
PROFILE_TASK="${PROFILE_TASK:-}"
PROFILE_MIXED_PRECISION="${PROFILE_MIXED_PRECISION:-bf16}"
PROFILE_REPORT_TO="${PROFILE_REPORT_TO:-tensorboard}"
PROFILE_ALLOW_TF32="${PROFILE_ALLOW_TF32:-true}"
PROFILE_MODE="${PROFILE_MODE:-time}"
PROFILE_TORCH_PROFILER="${PROFILE_TORCH_PROFILER:-true}"
PROFILE_WAIT="${PROFILE_WAIT:-1}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_ACTIVE="${PROFILE_ACTIVE:-3}"
PROFILE_REPEAT="${PROFILE_REPEAT:-1}"
PROFILE_SKIP_FIRST="${PROFILE_SKIP_FIRST:-0}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-true}"
PROFILE_MEMORY="${PROFILE_MEMORY:-true}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-true}"
PROFILE_WITH_FLOPS="${PROFILE_WITH_FLOPS:-true}"
PROFILE_USE_GZIP="${PROFILE_USE_GZIP:-false}"
PROFILE_CACHE_TO_RAM="${PROFILE_CACHE_TO_RAM:-false}"
PROFILE_CACHE_MODE="${PROFILE_CACHE_MODE:-lazy}"
PROFILE_CACHE_BACKEND="${PROFILE_CACHE_BACKEND:-local}"
PROFILE_CACHE_SOCKET="${PROFILE_CACHE_SOCKET:-}"
PROFILE_CACHE_MAX_EPISODES="${PROFILE_CACHE_MAX_EPISODES:-}"
PROFILE_CACHE_REPORT_INTERVAL="${PROFILE_CACHE_REPORT_INTERVAL:-1000}"
PROFILE_CACHE_BUILD_NUM_WORKERS="${PROFILE_CACHE_BUILD_NUM_WORKERS:-1}"

CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_STEM="${CONFIG_NAME%.*}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/profile_${CONFIG_STEM}_${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/profile.log"
ENV_FILE="${RUN_DIR}/environment.txt"
CMD_FILE="${RUN_DIR}/command.txt"
CACHE_SERVER_LOG_FILE="${RUN_DIR}/episode_cache_server.log"
CACHE_SERVER_READY_FILE="${RUN_DIR}/episode_cache_server.ready"
if [ -z "$PROFILE_CACHE_SOCKET" ]; then
    PROFILE_CACHE_SOCKET="${RUN_DIR}/episode_cache.sock"
fi

mkdir -p "$RUN_DIR"

{
    echo "timestamp: ${TIMESTAMP}"
    echo "host: $(hostname)"
    echo "pwd: $(pwd)"
    echo "config: ${CONFIG_PATH}"
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
    echo "nvidia-smi:"
    nvidia-smi 2>/dev/null || true
    echo
    echo "accelerate:"
    accelerate env 2>/dev/null || true
} > "$ENV_FILE"

HYDRA_OVERRIDES=(
    "_target_=drrm.trainer.profile.profile"
    "output_dir=${RUN_DIR}/checkpoints"
    "logging_dir=logs"
    "report_to=${PROFILE_REPORT_TO}"
    "mixed_precision=${PROFILE_MIXED_PRECISION}"
    "allow_tf32=${PROFILE_ALLOW_TF32}"
    "num_train_epochs=${PROFILE_EPOCHS}"
    "+profile.enabled=${PROFILE_TORCH_PROFILER}"
    "+profile.trace_dir=${RUN_DIR}/torch_profiler"
    "+profile.wait=${PROFILE_WAIT}"
    "+profile.warmup=${PROFILE_WARMUP}"
    "+profile.active=${PROFILE_ACTIVE}"
    "+profile.repeat=${PROFILE_REPEAT}"
    "+profile.skip_first=${PROFILE_SKIP_FIRST}"
    "+profile.record_shapes=${PROFILE_RECORD_SHAPES}"
    "+profile.profile_memory=${PROFILE_MEMORY}"
    "+profile.with_stack=${PROFILE_WITH_STACK}"
    "+profile.with_flops=${PROFILE_WITH_FLOPS}"
    "+profile.use_gzip=${PROFILE_USE_GZIP}"
    "+profile.skip_final_save=true"
    "checkpointing_period=1000000000"
    "val_period=-1"
    "num_val_batches=1"
)

if [ -n "$PROFILE_BATCH_SIZE" ]; then
    HYDRA_OVERRIDES+=("train_batch_size=${PROFILE_BATCH_SIZE}")
fi
if [ -n "$PROFILE_VAL_BATCH_SIZE" ]; then
    HYDRA_OVERRIDES+=("val_batch_size=${PROFILE_VAL_BATCH_SIZE}")
fi
if [ -n "$PROFILE_WORKERS" ]; then
    HYDRA_OVERRIDES+=("dataloader_num_workers=${PROFILE_WORKERS}")
fi
if [ -n "$PROFILE_TASK" ]; then
    HYDRA_OVERRIDES+=("train_dataset.task=${PROFILE_TASK}")
fi
if [ -n "$PROFILE_DEMO" ]; then
    HYDRA_OVERRIDES+=("train_dataset.demo=${PROFILE_DEMO}")
fi
if [ "$PROFILE_CACHE_TO_RAM" = "1" ] || [ "$PROFILE_CACHE_TO_RAM" = "true" ]; then
    HYDRA_OVERRIDES+=("++train_dataset.cache_to_ram=true")
    HYDRA_OVERRIDES+=("++train_dataset.cache_mode=${PROFILE_CACHE_MODE}")
    HYDRA_OVERRIDES+=("++train_dataset.cache_backend=${PROFILE_CACHE_BACKEND}")
    HYDRA_OVERRIDES+=("++train_dataset.cache_report_interval=${PROFILE_CACHE_REPORT_INTERVAL}")
    if [ "$PROFILE_CACHE_BACKEND" = "shm" ]; then
        HYDRA_OVERRIDES+=("++train_dataset.cache_socket=${PROFILE_CACHE_SOCKET}")
    fi
    if [ -n "$PROFILE_CACHE_MAX_EPISODES" ]; then
        HYDRA_OVERRIDES+=("++train_dataset.cache_max_episodes=${PROFILE_CACHE_MAX_EPISODES}")
    fi
fi

CMD=(
    accelerate launch
    --config_file configs/accelerate_config.yaml
    main.py
    "--config-path=${CONFIG_PATH%/*}"
    "--config-name=${CONFIG_NAME}"
    "${HYDRA_OVERRIDES[@]}"
    "$@"
)

printf '%q ' "${CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

echo "Profile output: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Episode cache server log: ${CACHE_SERVER_LOG_FILE}"
echo "Environment: ${ENV_FILE}"
echo "Command: ${CMD_FILE}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"


CACHE_SERVER_PID=""
cleanup_cache_server() {
    stop_shm_episode_cache_server "$PROFILE_CACHE_SOCKET" "$CACHE_SERVER_PID"
}
trap cleanup_cache_server EXIT

if { [ "$PROFILE_CACHE_TO_RAM" = "1" ] || [ "$PROFILE_CACHE_TO_RAM" = "true" ]; } && [ "$PROFILE_CACHE_BACKEND" = "shm" ]; then
    start_shm_episode_cache_server \
        "$CONFIG_PATH" \
        "$CONFIG_NAME" \
        "$PROFILE_CACHE_SOCKET" \
        "$CACHE_SERVER_LOG_FILE" \
        "$CACHE_SERVER_READY_FILE" \
        "${RUN_DIR}/episode_cache_server_command.txt" \
        "$PROFILE_CACHE_REPORT_INTERVAL" \
        "$PROFILE_CACHE_BUILD_NUM_WORKERS" \
        "$PROFILE_CACHE_MAX_EPISODES" \
        "$PROFILE_TASK" \
        "$PROFILE_DEMO"
    CACHE_SERVER_PID="$EPISODE_CACHE_SERVER_PID"
fi

SECONDS=0
set +e
if [ "$PROFILE_MODE" = "nsys" ] && command -v nsys >/dev/null 2>&1; then
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --output="${RUN_DIR}/nsys_report" \
        --force-overwrite=true \
        "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    STATUS=${PIPESTATUS[0]}
else
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
    elif command -v gtime >/dev/null 2>&1; then
        gtime -v "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
    else
        echo "Warning: GNU time not found; running without detailed /usr/bin/time metrics." | tee "$LOG_FILE"
        "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
        STATUS=${PIPESTATUS[0]}
    fi
fi
set -e

{
    echo "exit_status: ${STATUS}"
    echo "elapsed_seconds: ${SECONDS}"
    echo "run_dir: ${RUN_DIR}"
} | tee "${RUN_DIR}/summary.txt"

exit "$STATUS"
