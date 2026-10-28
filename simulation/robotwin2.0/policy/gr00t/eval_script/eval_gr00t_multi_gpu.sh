#!/usr/bin/env bash
# RoboTwin pi0.5 closed-loop eval, N GPUs, one self-contained worker per GPU.
#
# Each GPU gets its OWN openpi websocket server (own port) and its OWN disjoint
# slice of the pinned seed list, and writes its own result directory.  Nothing is
# shared between workers, so there is no concurrent-request state to reason about
# in the server -- which is the whole reason this is not script/eval_policy_parallel.py
# (that one points every worker at ONE host:port and steals work from a shared
# counter).
#
# Reproducibility: a scene and its inference noise are functions of ``now_seed``
# alone (``ep_num`` only reaches save paths), so which GPU ran episode k cannot
# change its outcome.  That is the same property the existing pinned-seed pairing
# across separate cloud jobs already depends on.
#
# Layout (the nesting matters -- see merge_multi_gpu.py):
#   <result_root>/<task>/gr00t/<config>/<arm>/<stamp>/           merged _eval_log/_eval_seeds/_result
#   <result_root>/<task>/gr00t/<config>/<arm>/<stamp>/1gpu/ ...  per-GPU output, videos included
#   <result_root>/<task>/gr00t/<config>/<arm>/<stamp>/_seeds_1gpu.txt ...  the split, as provenance
#
# **Staging lives next to the final directory, not in /tmp.**  The first version
# staged in /tmp: a preempted or crashed arm then left nothing at all -- no
# videos, no per-GPU logs, nothing to post-mortem -- while the serial path had
# always written straight to shared storage.  Same-filesystem staging also makes
# the collect step a rename instead of a 50 MB copy, and lets you watch the
# videos and the running success count while the arm is still going.
#
# Usage:  bash eval_gr00t_multi_gpu.sh <task> <task_config> <ckpt_setting> <seed> <test_num> <name:schedule>
# Env:    GR00T_GPUS (default 4), GR00T_PORT_BASE (5555), ROBOTWIN_SEED_LIST (required),
#         GR00T_PROGRESS_SECS (default 120), EVAL_RESULT_ROOT, plus what run_eval.sh needs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# GR00T takes the schedule as raw tau nodes (GR00T_TAU_NODES); there is no
# wire-token indirection to resolve, so the pi05 wire resolver has no counterpart here.
RUN_EVAL_SH="${RUN_EVAL_SH:-${GR00T_DIR}/run_eval.sh}"
MERGE_PY="${SCRIPT_DIR}/merge_multi_gpu.py"
STATUS_SH="${SCRIPT_DIR}/status_multi_gpu.sh"

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 <task> <task_config> <ckpt_setting> <seed> <test_num> <name:schedule>" >&2
  exit 2
fi
TASK_NAME="$1"; TASK_CONFIG="$2"; CKPT_SETTING="$3"; SEED_ARG="$4"; TEST_NUM="$5"; SPEC="$6"

RUN_NAME="${SPEC%%:*}"
SCHEDULE="${SPEC#*:}"
if [ -z "$RUN_NAME" ] || [ -z "$SCHEDULE" ]; then
  echo "Bad run spec (want name:schedule): $SPEC" >&2; exit 2
fi
# Validate the tau nodes here rather than discovering it inside a worker: strictly
# increasing, starts at 0, ends at 1.  "uniform<B>" is expanded to B equal steps.
SCHEDULE="$(python3 - "$SCHEDULE" <<'PYV'
import sys
s = sys.argv[1].strip()
if s.startswith("uniform"):
    B = int(s[len("uniform"):])
    nodes = [i / B for i in range(B + 1)]
else:
    nodes = [float(v) for v in s.replace(" ", "").split(",") if v]
if len(nodes) < 2 or abs(nodes[0]) > 1e-9 or abs(nodes[-1] - 1.0) > 1e-9:
    raise SystemExit(f"bad schedule {s!r}: must start at 0 and end at 1")
if any(b <= a for a, b in zip(nodes, nodes[1:])):
    raise SystemExit(f"bad schedule {s!r}: must be strictly increasing")
print(",".join(repr(float(v)) for v in nodes))
PYV
)" || { echo "Bad schedule: ${SPEC#*:}" >&2; exit 2; }

NGPU="${GR00T_GPUS:-4}"
# Physical device of worker 0.  Default 0 keeps every existing caller bit-identical;
# it exists so two arms can share one 8-GPU box (offset 0 and 4) without either of
# them touching the other's cards.  The worker *index* (and so the seed slice, the
# port and the result part name) stays 0..NGPU-1 -- only the device moves.
GPU_OFFSET="${GR00T_GPU_OFFSET:-0}"
PORT_BASE="${GR00T_PORT_BASE:-5555}"
PROGRESS_SECS="${GR00T_PROGRESS_SECS:-120}"
RESULT_ROOT="${EVAL_RESULT_ROOT:-/DRoboticsAILab/users/licheng.yang/robotwin2.0/result}"
SEED_LIST="${ROBOTWIN_SEED_LIST:?ROBOTWIN_SEED_LIST is required (pinned seeds)}"
# Relocate the result tree without changing the loaded task_config yml.
RESULT_CONFIG="${EVAL_RESULT_CONFIG:-${TASK_CONFIG}}"
export EVAL_RESULT_CONFIG="$RESULT_CONFIG"

TEST_NUM_TOTAL="$TEST_NUM"   # the arm's episode count, before the per-GPU split
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
FINAL="${RESULT_ROOT}/${TASK_NAME}/gr00t/${RESULT_CONFIG}/${RUN_NAME}/${STAMP}"
STAGE="${FINAL}/_staging"
mkdir -p "${STAGE}"
MASTER_LOG="${FINAL}/_launcher.log"

# --- split the pinned seeds into NGPU contiguous, disjoint slices ------------------
python3 - "$SEED_LIST" "$TEST_NUM" "$NGPU" "$FINAL" <<'PY' | tee -a "$MASTER_LOG"
import sys, os
path, total, n, final = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
raw = open(path, encoding="utf-8").read() if os.path.isfile(path) else path
seeds = [s.strip() for s in raw.replace("\n", ",").split(",") if s.strip()][:total]
if len(seeds) != total:
    sys.exit(f"[split] seed list has {len(seeds)} seeds but test_num={total}")
base, rem = divmod(total, n)
i = 0
for k in range(n):
    take = base + (1 if k < rem else 0)
    chunk = seeds[i:i + take]; i += take
    # Written next to the result, not into a temp dir: this is the receipt that the
    # four slices were disjoint and covered the pinned list exactly.
    with open(os.path.join(final, f"_seeds_{k + 1}gpu.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(chunk) + "\n")
    print(f"[split] {k + 1}gpu: {take} seeds  {chunk[0]}..{chunk[-1]}")
PY

{
  echo "[mgpu] task=${TASK_NAME} run=${RUN_NAME} schedule=${SCHEDULE}"
  echo "[mgpu] gpus=${NGPU} gpu_offset=${GPU_OFFSET} port_base=${PORT_BASE} test_num=${TEST_NUM} seed=${SEED_ARG}"
  echo "[mgpu] final=${FINAL}"
  nvidia-smi -L 2>/dev/null || true
} | tee -a "$MASTER_LOG"

# --- one self-contained worker per GPU --------------------------------------------
PIDS=(); PARTS=()
for ((k = 0; k < NGPU; k++)); do
  part="$((k + 1))gpu"
  PARTS+=("$part")
  mkdir -p "${STAGE}/stage${k}"
  wlog="${FINAL}/_worker_${part}.log"
  (
    export GPU_ID="$((GPU_OFFSET + k))"
    export KRAKEN_EGL_DEVICE="$((GPU_OFFSET + k))"
    export GR00T_PORT=$((PORT_BASE + k))
    export GR00T_TAU_NODES="$SCHEDULE"
    # Pass the shared-storage interpreter through to serve.sh: the tree's own
    # .venv symlinks /root/miniconda3, which no 8-GPU worker has.
    export GR00T_PY="${GR00T_PY:-/data/vepfs/licheng.yang/pythons/miniconda3/bin/python3.12}"
    export PROBE_PY="${PROBE_PY:-${GR00T_PY}}"
    # Client vs server run on different interpreters, and neither venv symlink resolves on
    # an 8-GPU worker (both point into /root/miniconda3, which is container-local):
    #   client  RoboTwin + sapien -> /usr/bin/python3.10 with the venv site-packages on
    #           PYTHONPATH (the site-packages themselves are on EFS)
    #   server  GR00T py3.12      -> GR00T_PY above
    # This is the same shape as the pi05 arms, which resolve to /usr/bin/python3.10 too.
    export PYTHON="${ROBOTWIN_PY:-/usr/bin/python3.10}"
    export ROBOTWIN_VENV_SITE="${ROBOTWIN_ROOT:-/data/vepfs/licheng.yang/DRRM-VODPP/simulation/robotwin2.0}/.venv/lib/python3.10/site-packages"
    # Editable installs (curobo, gtinverse, vggt) live as .pth files that only a real
    # venv startup expands; a hand-set PYTHONPATH does not read them, and the eval dies
    # much later on "cannot import name CuroboPlanner".  Expand them here.
    export PYTHONPATH="${ROBOTWIN_VENV_SITE}:$(cat "${ROBOTWIN_VENV_SITE}"/__editable__*.pth 2>/dev/null | grep -E '^/' | paste -sd: -)${PYTHONPATH:+:${PYTHONPATH}}"
    export ROBOTWIN_SEED_LIST="${FINAL}/_seeds_${part}.txt"
    export TEST_NUM="$(wc -l < "${FINAL}/_seeds_${part}.txt")"
    # **The instruction pool must stay the arm's size, not the worker's.**
    # eval_policy.py used ``test_num`` for two things: how many episodes this
    # process runs, and the size of the instruction pool that
    # ``np.random.choice`` draws the policy's prompt from.  A worker running 25
    # of 100 episodes would draw from a 25-sentence pool and get a DIFFERENT
    # prompt for the same seed -- making every 4-GPU arm incomparable with every
    # serial arm ever recorded, silently.  Verified 2026-08-19: same seed,
    # pool 25 -> "Move red block and green block to the center...",
    # pool 100 -> "Bring red block and green block to the center...".
    export ROBOTWIN_DESC_NUM="$TEST_NUM_TOTAL"
    export EVAL_RESULT_ROOT="${STAGE}/stage${k}"
    export EVAL_RUN_NAME="$RUN_NAME"
    unset GR00T_REPLAN_STEPS
    bash "$RUN_EVAL_SH" "$TASK_NAME" "$TASK_CONFIG" "$CKPT_SETTING" "$SEED_ARG" "$((GPU_OFFSET + k))"
  ) >"$wlog" 2>&1 &
  PIDS+=("$!")
  echo "[mgpu] launched ${part} pid=${PIDS[-1]} gpu=$((GPU_OFFSET + k)) port=$((PORT_BASE + k)) -> ${wlog}" | tee -a "$MASTER_LOG"
done

# --- running success/fail count, so the arm is watchable while it runs -------------
# stdout/stderr go to /dev/null on purpose: the ticker's ``sleep`` inherits this
# script's stdout, and a ``sleep`` that outlives the kill below would hold the
# pipe open -- anything piping this script's output (a launcher, a `| tee`) then
# waits for an EOF that never comes.  Progress goes to the master log by name.
(
  while sleep "$PROGRESS_SECS"; do
    bash "$STATUS_SH" "$FINAL" --oneline >>"$MASTER_LOG" 2>/dev/null || true
  done
) >/dev/null 2>&1 &
TICKER=$!
stop_ticker() {
  [ -n "${TICKER:-}" ] || return 0
  kill "$TICKER" 2>/dev/null || true
  wait "$TICKER" 2>/dev/null || true
  TICKER=""
}
trap stop_ticker EXIT

FAIL=0
for idx in "${!PIDS[@]}"; do
  set +e; wait "${PIDS[$idx]}"; rc=$?; set -e
  part="${PARTS[$idx]}"
  if [ "$rc" -ne 0 ] || ! grep -q "Data has been saved to" "${FINAL}/_worker_${part}.log"; then
    echo "[mgpu] FAIL ${part} rc=${rc}" | tee -a "$MASTER_LOG"
    tail -n 25 "${FINAL}/_worker_${part}.log" | tee -a "$MASTER_LOG" || true
    FAIL=1
  else
    echo "[mgpu] OK   ${part}" | tee -a "$MASTER_LOG"
  fi
done
stop_ticker

if [ "$FAIL" -ne 0 ]; then
  echo "[mgpu] a worker failed; NOT merging (a partial merge would look like a real run)." \
    | tee -a "$MASTER_LOG"
  echo "[mgpu] staging kept for post-mortem: ${STAGE}" | tee -a "$MASTER_LOG"
  exit 1
fi

# --- collect the per-GPU directories under <stamp>/ (same filesystem: a rename) ----
for ((k = 0; k < NGPU; k++)); do
  part="$((k + 1))gpu"
  src="$(find "${STAGE}/stage${k}/${TASK_NAME}/gr00t/${RESULT_CONFIG}/${RUN_NAME}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
  if [ -z "$src" ]; then
    echo "[mgpu] no output dir for ${part}; staging kept: ${STAGE}" | tee -a "$MASTER_LOG"; exit 1
  fi
  mv "$src" "${FINAL}/${part}"
  echo "[mgpu] ${part} <- ${src}  ($(ls "${FINAL}/${part}"/*.mp4 2>/dev/null | wc -l) videos)" | tee -a "$MASTER_LOG"
done
rm -rf "${STAGE}"

python3 "$MERGE_PY" --stamp-dir "$FINAL" --parts "${PARTS[@]}" --expect "$TEST_NUM" --allow-short | tee -a "$MASTER_LOG"
rc=${PIPESTATUS[0]}
bash "$STATUS_SH" "$FINAL" | tee -a "$MASTER_LOG" || true
echo "[mgpu] done rc=${rc} -> ${FINAL}" | tee -a "$MASTER_LOG"
exit "$rc"
