#!/usr/bin/env bash
# Running success/fail count for one multi-GPU arm, mid-run or finished.
#
#   bash status_multi_gpu.sh <result_root>/<task>/cosmos/<config>/<arm>/<stamp>/
#   bash status_multi_gpu.sh <stamp_dir> --oneline      # one line, for the launcher log
#
# Reads whichever per-GPU logs exist: the staged ones while the arm is running,
# the collected ``Ngpu/`` ones once it has finished.  Counting from the logs
# rather than from a status file means it is never out of date.
set -uo pipefail

DIR="${1:?usage: status_multi_gpu.sh <stamp_dir> [--oneline]}"
ONELINE=0
[ "${2:-}" = "--oneline" ] && ONELINE=1

total_n=0; total_s=0; total_f=0
parts=""
for k in 1 2 3 4 5 6 7 8; do
  log="${DIR}/${k}gpu/_eval_log.txt"
  if [ ! -f "$log" ]; then
    # still running: the worker writes into the staging tree
    log="$(ls "${DIR}/_staging/stage$((k - 1))"/*/cosmos/*/*/*/_eval_log.txt 2>/dev/null | head -1)"
  fi
  [ -z "$log" ] || [ ! -f "$log" ] && { [ -d "${DIR}/${k}gpu" ] || [ -d "${DIR}/_staging/stage$((k - 1))" ] || continue; }
  [ -f "$log" ] || continue
  want=0
  [ -f "${DIR}/_seeds_${k}gpu.txt" ] && want=$(wc -l < "${DIR}/_seeds_${k}gpu.txt")
  n=$(grep -c 'result:' "$log" 2>/dev/null || echo 0)
  s=$(grep -c 'result: Success' "$log" 2>/dev/null || echo 0)
  f=$((n - s))
  total_n=$((total_n + n)); total_s=$((total_s + s)); total_f=$((total_f + f))
  parts="${parts}${k}gpu ${n}/${want} (成功 ${s} 失败 ${f}) | "
  [ "$ONELINE" -eq 0 ] && printf "  %dgpu  已跑 %3d/%-3d  成功 %3d  失败 %3d\n" "$k" "$n" "$want" "$s" "$f"
done

if [ "$total_n" -eq 0 ]; then
  echo "[status] 还没有 episode 完成 ($DIR)"
  exit 0
fi
rate=$(awk "BEGIN{printf \"%.3f\", $total_s/$total_n}")
if [ "$ONELINE" -eq 1 ]; then
  echo "[progress] ${parts}合计 ${total_n} 集: 成功 ${total_s} 失败 ${total_f} (成功率 ${rate})"
else
  printf "  ------\n  合计   已跑 %3d      成功 %3d  失败 %3d   成功率 %s\n" \
    "$total_n" "$total_s" "$total_f" "$rate"
fi
