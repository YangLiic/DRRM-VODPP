"""Collect the success rate of every Cosmos eval run under a result root.

RoboTwin writes one ``_eval_log.txt`` per (task, policy, task_config, run, timestamp);
the header of each carries the success rate and the per-episode lines carry the
inference latency. This prints one row per run, newest last.

    python summarize.py [result_root] [--run-name cosmos3_edge_robotwin]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_RESULT_ROOT = "/DRoboticsAILab/users/licheng.yang/robotwin2.0/result"
_SUCCESS_RE = re.compile(r"Success Rate:\s*([0-9.]+)\s*\((\d+)/(\d+)\)")
_INFER_MS_RE = re.compile(r"infer_ms_mean:\s*([0-9.]+)")
_STEPS_RE = re.compile(r"action_cnt:\s*(\d+)")


def _parse(log_path: Path) -> dict | None:
    text = log_path.read_text(errors="replace")
    match = _SUCCESS_RE.search(text)
    if match is None:
        return None  # run still in flight: the header is written at the end
    latencies = [float(v) for v in _INFER_MS_RE.findall(text)]
    steps = [int(v) for v in _STEPS_RE.findall(text)]
    # .../<task>/<policy>/<task_config>/<run_name>/<timestamp>/_eval_log.txt
    parts = log_path.parts
    return {
        "task": parts[-6],
        "policy": parts[-5],
        "task_config": parts[-4],
        "run": parts[-3],
        "timestamp": parts[-2],
        "rate": float(match.group(1)),
        "success": int(match.group(2)),
        "total": int(match.group(3)),
        "infer_ms_mean": sum(latencies) / len(latencies) if latencies else None,
        "steps_mean": sum(steps) / len(steps) if steps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", nargs="?", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--policy", default="cosmos")
    parser.add_argument("--run-name", default=None, help="filter to one run subdirectory")
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = []
    for log_path in root.glob(f"*/{args.policy}/*/*/*/_eval_log.txt"):
        row = _parse(log_path)
        if row is None:
            continue
        if args.run_name and row["run"] != args.run_name:
            continue
        rows.append(row)

    if not rows:
        print(f"no finished {args.policy} runs under {root}")
        return

    rows.sort(key=lambda r: (r["timestamp"], r["task"]))
    print(f"{'timestamp':<21} {'task':<22} {'config':<16} {'run':<24} {'success':>9} {'rate':>7} {'infer_ms':>9} {'steps':>7}")
    print("-" * 121)
    total_success = total_episodes = 0
    for row in rows:
        latency = f"{row['infer_ms_mean']:.0f}" if row["infer_ms_mean"] is not None else "-"
        steps = f"{row['steps_mean']:.0f}" if row["steps_mean"] is not None else "-"
        print(
            f"{row['timestamp']:<21} {row['task']:<22} {row['task_config']:<16} {row['run']:<24} "
            f"{row['success']:>4}/{row['total']:<4} {row['rate']*100:>6.1f}% {latency:>9} {steps:>7}"
        )
        total_success += row["success"]
        total_episodes += row["total"]
    print("-" * 121)
    print(f"{'TOTAL':<21} {'':<22} {'':<16} {'':<24} {total_success:>4}/{total_episodes:<4} "
          f"{total_success / total_episodes * 100:>6.1f}%")


if __name__ == "__main__":
    main()
