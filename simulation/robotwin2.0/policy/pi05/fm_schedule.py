"""pi0.5 Flow-Matching inference tau schedules.

openpi / pi0.5 integrates noise→action with t: **1 → 0** (opposite of VODPP's 0→1).
Tokens and explicit knots in this module always use the pi0.5 direction.

Examples:
  uniform10          -> [1.0, 0.9, ..., 0.0]   (10 NFE, openpi default)
  uniform3           -> [1.0, 2/3, 1/3, 0.0]
  1,0.7,0.3,0        -> 3 NFE custom knots

Do NOT pass VODPP / TrustEuler 0→1 grids here. Convert first: tau_pi = 1 - tau_vodpp.

taures schedules (gtinverse docs/TAU_RESIDUAL_PLAN.md section 4) append two
scalars to the token, both **canonical** (0=noise, 1=action) exactly as the
schedule-grid artifact writes them:

  1,0.75,0.5,0@tau_star=0.5@q_last=0.55

The suffix passes through resolution untouched; serve_policy.py is the single
place it is converted to the native convention.  Tokens without the suffix
resolve exactly as before.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Sequence, Union

ScheduleInput = Union[str, Sequence[float], None]


def split_taures_suffix(token: str) -> tuple[str, Optional[dict]]:
    """Split ``"<knots>@tau_star=..@q_last=.."`` into (knots token, suffix dict).

    Values are validated for range and mutual presence but *not* converted:
    they stay canonical, and serve_policy.py owns the one conversion to the
    native direction.  Tokens without ``@`` return (token, None).
    """
    parts = token.split("@")
    if len(parts) == 1:
        return token, None
    extras: dict = {}
    for piece in parts[1:]:
        key, _, value = piece.partition("=")
        key = key.strip().lower()
        if key not in ("tau_star", "q_last") or not value.strip():
            raise ValueError(f"unknown taures suffix {piece!r}")
        extras[key] = float(value)
    if set(extras) != {"tau_star", "q_last"}:
        raise ValueError("taures suffix needs both tau_star and q_last")
    for key, value in extras.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{key} must lie in (0, 1), got {value}")
    return parts[0], extras


def format_taures_suffix(extras: Optional[dict]) -> str:
    if not extras:
        return ""
    return f"@tau_star={extras['tau_star']:g}@q_last={extras['q_last']:g}"


def resolve_pi05_tau_schedule(
    schedule: ScheduleInput = None,
    *,
    num_steps: Optional[int] = None,
) -> List[float]:
    """Resolve knotted tau grid for pi0.5 Euler ODE (strictly decreasing 1→0).

    Args:
        schedule:
            - None: uniform with ``num_steps`` (default 10).
            - "uniform" / "uniformN": linspace(1, 0, N+1).
            - "1,0.7,0.3,0" or [1, 0.7, 0.3, 0]: explicit knots.
        num_steps: used when schedule is None / bare "uniform".

    Returns:
        Strictly decreasing knot list; NFE = len(knots) - 1.
    """
    if schedule is None:
        n = int(num_steps if num_steps is not None else 10)
        knots = [1.0 - i / float(n) for i in range(n + 1)]
    elif isinstance(schedule, str):
        s = schedule.strip().lower().replace(" ", "")
        m = re.fullmatch(r"uniform(\d*)", s)
        if m is not None:
            n_str = m.group(1)
            n = int(n_str) if n_str else int(num_steps if num_steps is not None else 10)
            knots = [1.0 - i / float(n) for i in range(n + 1)]
        else:
            s = s.strip("[]()")
            knots = [float(x) for x in s.split(",") if x != ""]
    else:
        knots = [float(x) for x in schedule]

    if len(knots) < 2:
        raise ValueError(f"tau schedule needs >= 2 knots (got {knots}); NFE = len(knots)-1")

    diffs = [knots[i + 1] - knots[i] for i in range(len(knots) - 1)]
    if any(abs(d) < 1e-12 for d in diffs):
        raise ValueError(f"tau schedule has zero-width interval: {knots}")
    if any(d >= 0 for d in diffs):
        raise ValueError(
            "pi0.5 tau schedule must be strictly decreasing (1→0 noise→action); "
            f"got {knots}. If you copied a VODPP/TrustEuler grid, use tau_pi = 1 - tau_vodpp."
        )
    for t in knots:
        if t < -1e-6 or t > 1.0 + 1e-6:
            raise ValueError(f"tau knot out of [0, 1]: {knots}")

    return knots


def format_knots(knots: Sequence[float], *, precision: int = 6) -> str:
    """Comma-separated knots for CLI / serve_policy --fm-tau-schedule."""

    def _fmt(x: float) -> str:
        s = f"{x:.{precision}f}".rstrip("0").rstrip(".")
        return s if s else "0"

    return ",".join(_fmt(float(t)) for t in knots)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "token",
        help='Schedule token: uniformN or "1,0.7,0.3,0"',
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Fallback N for bare 'uniform' (default 10).",
    )
    parser.add_argument(
        "--nfe",
        action="store_true",
        help="Print NFE only.",
    )
    args = parser.parse_args(argv)
    token, extras = split_taures_suffix(args.token)
    knots = resolve_pi05_tau_schedule(token, num_steps=args.num_steps)
    if args.nfe:
        print(len(knots) - 1)
    else:
        print(format_knots(knots) + format_taures_suffix(extras))
    return 0


if __name__ == "__main__":
    sys.exit(main())
