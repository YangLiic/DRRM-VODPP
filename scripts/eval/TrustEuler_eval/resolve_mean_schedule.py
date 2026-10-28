#!/usr/bin/env python3
"""Resolve a schedule token to an fm_tau_schedule string.

Tokens:
  mean3 / mean5 / mean8 / mean10  -> read tau_nodes from TrustEuler mean_grid.json
  uniformN                         -> passed through as-is
  0,0.25,0.5,1                     -> passed through as-is
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_MEAN_RE = re.compile(r"^mean(\d+)$", re.IGNORECASE)


def format_tau_nodes(nodes: list[float]) -> str:
    parts: list[str] = []
    for value in nodes:
        text = f"{float(value):.8f}".rstrip("0").rstrip(".")
        if text in {"-0", ""}:
            text = "0"
        parts.append(text)
    return ",".join(parts)


def resolve_schedule(token: str, *, mean_grid: Path | None) -> str:
    token = token.strip()
    if not token:
        raise ValueError("empty schedule token")

    match = _MEAN_RE.fullmatch(token)
    if match is None:
        return token

    budget = int(match.group(1))
    if mean_grid is None:
        raise ValueError(f"{token} requires --mean-grid")
    if not mean_grid.is_file():
        raise FileNotFoundError(f"mean grid missing: {mean_grid}")

    artifact = json.loads(mean_grid.read_text(encoding="utf-8"))
    schedules = artifact.get("schedules")
    if not isinstance(schedules, dict) or str(budget) not in schedules:
        available = sorted(schedules.keys()) if isinstance(schedules, dict) else []
        raise KeyError(f"budget {budget} absent from {mean_grid}; available={available}")
    nodes = schedules[str(budget)]["tau_nodes"]
    if not isinstance(nodes, list) or len(nodes) != budget + 1:
        raise ValueError(f"invalid tau_nodes for budget {budget} in {mean_grid}")
    return format_tau_nodes([float(x) for x in nodes])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token")
    parser.add_argument("--mean-grid", type=Path, default=None)
    args = parser.parse_args()
    try:
        print(resolve_schedule(args.token, mean_grid=args.mean_grid))
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
