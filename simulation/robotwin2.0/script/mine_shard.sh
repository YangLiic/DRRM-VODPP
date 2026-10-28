#!/bin/bash
# Mine demo_clean seed lists for one shard of tasks, one task per GPU in parallel.
# Mining needs only the curobo expert (no policy), so a shard of 8 runs concurrently
# on an 8-GPU box; the whole remaining-task set finishes in one task's wall-clock
# instead of twenty-five.
# usage: mine_shard.sh <task> [task ...]
set -u
RW=/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0
cd "$RW" || exit 1
OUT=/DRoboticsAILab/users/licheng.yang/robotwin2.0/seedlists
LOGD=$OUT/_mining_logs
mkdir -p "$LOGD"
# One task per GPU, wrapping when the shard is longer than the box: assigning
# CUDA_VISIBLE_DEVICES=8 on an 8-GPU machine does not fail loudly -- CUDA init fails and
# curobo's import dies with "cannot import name 'CuroboPlanner'", which reads like a code
# problem rather than an off-by-one (measured 2026-08-30 on stack_bowls_two).
NV=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$NV" -ge 1 ] || { echo "no GPUs visible"; exit 1; }
echo "[shard] $NV GPU(s), $# task(s)"
pids=(); i=0
for t in "$@"; do
  f="$OUT/${t}_demo_clean.txt"
  if [ -f "$f" ] && [ "$(wc -w < "$f")" -ge 100 ]; then
    echo "[skip] $t has $(wc -w < "$f")"; continue
  fi
  echo "[gpu $(( i % NV ))] START $t"
  CUDA_VISIBLE_DEVICES=$(( i % NV )) bash "$RW/script/mine_eval_seeds_cloud.sh" \
      --task_name "$t" --task_config demo_clean --test_num 100 --max_walk 6000 \
      --out "$f" > "$LOGD/${t}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== shard done, nonzero-exit=$fail ==="
for t in "$@"; do
  f="$OUT/${t}_demo_clean.txt"; n=0; [ -f "$f" ] && n=$(wc -w < "$f")
  printf "  %-30s %s\n" "$t" "$n"
done
