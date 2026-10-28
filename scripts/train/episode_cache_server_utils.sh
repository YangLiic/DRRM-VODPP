#!/usr/bin/env bash

# Shared helpers for launching the rank-shared DRRM shm episode cache server.
# Callers own the run directory and Hydra overrides; this file owns only server lifecycle.

EPISODE_CACHE_SERVER_PID="${EPISODE_CACHE_SERVER_PID:-}"

start_shm_episode_cache_server() {
    local config_path="$1"
    local config_name="$2"
    local socket_path="$3"
    local log_file="$4"
    local ready_file="$5"
    local command_file="$6"
    local report_interval="$7"
    local build_num_workers="$8"
    local max_episodes="$9"
    local task="${10:-}"
    local demo="${11:-}"
    local extra_overrides=("${@:12}")
    local ready_timeout="${EPISODE_CACHE_READY_TIMEOUT:-7200}"

    local cache_server_cmd=(
        python -m drrm.datasets.shm_episode_cache
        --config-path "${config_path%/*}"
        --config-name "${config_name}"
        --socket "$socket_path"
        --cache-report-interval "$report_interval"
        --build-num-workers "$build_num_workers"
        --ready-file "$ready_file"
    )
    if [ -n "$max_episodes" ]; then
        cache_server_cmd+=(--cache-max-episodes "$max_episodes")
    fi
    if [ -n "$task" ]; then
        cache_server_cmd+=("train_dataset.task=${task}")
    fi
    if [ -n "$demo" ]; then
        cache_server_cmd+=("train_dataset.demo=${demo}")
    fi
    if [ ${#extra_overrides[@]} -gt 0 ]; then
        cache_server_cmd+=("${extra_overrides[@]}")
    fi

    printf '%q ' "${cache_server_cmd[@]}" > "$command_file"
    printf '\n' >> "$command_file"

    echo "Starting shm episode cache server..."
    "${cache_server_cmd[@]}" > "$log_file" 2>&1 &
    EPISODE_CACHE_SERVER_PID=$!

    for _ in $(seq 1 "$ready_timeout"); do
        if [ -f "$ready_file" ]; then
            echo "shm episode cache server ready: ${socket_path}"
            return 0
        fi
        if ! kill -0 "$EPISODE_CACHE_SERVER_PID" >/dev/null 2>&1; then
            echo "shm episode cache server exited before ready. See ${log_file}" >&2
            return 1
        fi
        sleep 1
    done

    echo "Timed out waiting for shm episode cache server. See ${log_file}" >&2
    return 1
}

stop_shm_episode_cache_server() {
    local socket_path="$1"
    local server_pid="${2:-$EPISODE_CACHE_SERVER_PID}"
    if [ -n "$server_pid" ] && kill -0 "$server_pid" >/dev/null 2>&1; then
        python - <<PY >/dev/null 2>&1 || true
from drrm.datasets.shm_episode_cache import ShmEpisodeCacheClient
ShmEpisodeCacheClient("$socket_path", timeout=5.0).stop()
PY
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
}
