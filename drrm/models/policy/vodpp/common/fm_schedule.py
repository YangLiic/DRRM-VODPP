"""Flow-Matching inference time (tau) schedules for VODPP.

VODPP linear FM uses path x_t = (1-t)*noise + t*action with t: 0 -> 1
(noise -> data), and Euler integrates with +dt * velocity. So a knot list
like [0, 0.25, 0.5, 1] is used as-is (3 NFE). If a model instead integrates
1 -> 0, pass invert=True to map knots via tau' = 1 - tau.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

ScheduleInput = Union[str, Sequence[float], None]

# A solved tau-residual schedule arrives as <knots>@tau_star=X@q_last=Y.
# The knots already carry the last step's *width* (their last interval is
# 1 - X), so the suffix adds exactly one thing: the tau at which that final
# step evaluates the network.  That is identity 2 -- X1 = x + (1-X)*v(x, Y) --
# which holds exactly for any (stride, query) pair, so a server that honours it
# needs no other change.  Without the suffix the schedule is a plain Euler grid
# and behaves exactly as before.
_LAST_STEP_RE = re.compile(
    r"@tau_star=(?P<tau_star>[-+0-9.eE]+)@q_last=(?P<q_last>[-+0-9.eE]+)\Z"
)


def resolve_inference_tau_schedule(
    schedule: ScheduleInput = None,
    num_inference_timesteps: Optional[int] = None,
    invert: bool = False,
) -> List[float]:
    """Backwards-compatible wrapper: knots only, last-step override dropped.

    Kept so that callers that never learned about the suffix keep working; a
    caller that *can* honour the decoupled last step must use
    resolve_inference_tau_plan instead, because silently dropping q_last
    here would run a different sampler than the one that was solved for.
    """
    return resolve_inference_tau_plan(
        schedule, num_inference_timesteps=num_inference_timesteps, invert=invert
    )[0]


def resolve_inference_tau_plan(
    schedule: ScheduleInput = None,
    num_inference_timesteps: Optional[int] = None,
    invert: bool = False,
) -> tuple:
    """Resolve knotted tau grid for Euler ODE.

    Args:
        schedule:
            - None: uniform grid with ``num_inference_timesteps`` steps
              (same as current VODPP default).
            - "uniform" / "uniform10" / "uniformN": linspace(0, 1, N+1).
            - sequence of floats, e.g. [0, 0.25, 0.5, 1]: NFE = len - 1.
            - string "0,0.25,0.5,1" or "[0, 0.25, 0.5, 1]".
        num_inference_timesteps: used when schedule is None / bare "uniform".
        invert: if True, replace each knot tau with (1 - tau). Use when the
            FM ODE runs from 1 -> 0 instead of VODPP's 0 -> 1.

    Returns:
        Monotonic knot list of length NFE+1 (including endpoints).
    """
    last_query = None
    if isinstance(schedule, str):
        match = _LAST_STEP_RE.search(schedule.strip().replace(" ", ""))
        if match is not None:
            schedule = schedule.strip().replace(" ", "")[: match.start()]
            last_query = float(match.group("q_last"))
            tau_star = float(match.group("tau_star"))

    if schedule is None:
        if num_inference_timesteps is None:
            raise ValueError("Need schedule or num_inference_timesteps")
        n = int(num_inference_timesteps)
        knots = [i / float(n) for i in range(n + 1)]
    elif isinstance(schedule, str):
        s = schedule.strip().lower().replace(" ", "")
        m = re.fullmatch(r"uniform(\d*)", s)
        if m is not None:
            n_str = m.group(1)
            n = int(n_str) if n_str else int(num_inference_timesteps or 10)
            knots = [i / float(n) for i in range(n + 1)]
        else:
            # "0,0.25,0.5,1" or "[0,0.25,0.5,1]"
            s = s.strip("[]()")
            knots = [float(x) for x in s.split(",") if x != ""]
    else:
        knots = [float(x) for x in schedule]

    if len(knots) < 2:
        raise ValueError(
            f"tau schedule needs >= 2 knots (got {knots}); NFE = len(knots)-1"
        )

    if invert:
        knots = [1.0 - t for t in knots]

    # Allow either increasing (0->1) or decreasing (1->0) grids.
    diffs = [knots[i + 1] - knots[i] for i in range(len(knots) - 1)]
    if any(abs(d) < 1e-12 for d in diffs):
        raise ValueError(f"tau schedule has zero-width interval: {knots}")
    signs = [d > 0 for d in diffs]
    if not (all(signs) or not any(signs)):
        raise ValueError(f"tau schedule must be monotonic: {knots}")

    if last_query is not None:
        # tau_star in the wire is the point the last step rides *from*, so
        # it must be the second-to-last knot.  A mismatch means the wire was
        # assembled from two different solves; running it would silently pair
        # one schedule's stride with another's query.
        if abs(knots[-2] - tau_star) > 1e-6:
            raise ValueError(
                f"wire says the last step rides from {tau_star}, but the knots "
                f"ride from {knots[-2]}: {knots}"
            )
        if invert:
            last_query = 1.0 - last_query
        if not 0.0 <= last_query <= 1.0:
            raise ValueError(f"q_last must lie in [0, 1], got {last_query}")

    return knots, last_query


def tau_to_timestep_bucket(tau: float, num_timestep_buckets: int) -> int:
    """Map continuous tau in [0, 1] to discrete training bucket index."""
    # Match training: (t * num_timestep_buckets).long(); clamp for tau==1.
    idx = int(tau * num_timestep_buckets)
    if idx >= num_timestep_buckets:
        idx = num_timestep_buckets - 1
    if idx < 0:
        idx = 0
    return idx
