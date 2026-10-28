"""Merge N per-GPU RoboTwin eval outputs into one arm-level result.

Each GPU ran a disjoint slice of the SAME pinned seed list with its own openpi
server, so the union is exactly the run a single GPU would have produced: the
scene and the inference noise are both derived from ``now_seed`` alone
(``ep_num`` only reaches save paths), which is the property the pinned-seed
pairing across separate cloud jobs already relies on.

What downstream actually reads (``tools/report_closed_loop.py::load``):
  ``<arm>/<stamp>/_eval_log.txt`` -- the ``FM tau schedule:`` header line and the
  per-seed ``seed: N, result: X`` lines, matched by regex, order-free.
It globs ``<arm>/*/`` ONE level deep, so the per-GPU directories must sit under
``<stamp>/`` rather than beside it; otherwise the reader picks ``4gpu`` and every
paired comparison silently runs on a quarter of the episodes.

**Short runs.**  A pinned seed can be valid to the expert at mining time and still
fail in eval mode -- deterministically for some tasks (``open_laptop`` 100026,
``grab_roller`` 100105) and intermittently for others (``open_microwave``, where six
arms each lost a different one to four seeds).  Refusing the merge protects the
pairing, but it also throws away every episode that *did* run: the arm is re-run from
zero for the sake of one seed.

``--allow-short`` keeps the work instead.  The result is written with the episodes
that landed, the seeds that did not are listed in ``_missing_seeds.txt`` beside it,
and the header records the shortfall so no reader can mistake it for a full run.
Repair is then a top-up (``exsched tools/topup_arms.py``): drop what any arm missed,
mine that many replacements, and run *only* the replacements on every arm -- which
restores a common seed set across arms at a fraction of the cost.  Without the flag
the behaviour is exactly as before, so nothing already wired up changes.
"""
import argparse
import os
import re
import sys
import time

LINE = re.compile(r"^【50*(\d+)】(seed:\s*(\d+),\s*result:\s*(\w+).*)$")
ANY = re.compile(r"^【\d+】(seed:\s*(\d+),\s*result:\s*(\w+).*)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp-dir", required=True, help="<arm>/<stamp>/, holds the Ngpu/ subdirs")
    ap.add_argument("--parts", nargs="+", required=True, help="Ngpu subdir names, in order")
    ap.add_argument("--expect", type=int, required=True, help="total episodes expected")
    ap.add_argument("--allow-short", action="store_true",
                    help="write the result even if some seeds produced no episode, and "
                         "record which ones in _missing_seeds.txt (default: refuse)")
    ap.add_argument("--min-fraction", type=float, default=0.8,
                    help="with --allow-short, still refuse below this fraction of --expect: "
                         "a run that lost a fifth of its scenes is a failure, not a shortfall")
    a = ap.parse_args()

    header, rows = [], []
    for part in a.parts:
        log = os.path.join(a.stamp_dir, part, "_eval_log.txt")
        if not os.path.isfile(log):
            print("[merge] MISSING " + log, file=sys.stderr)
            return 1
        with open(log, encoding="utf-8") as fh:
            lines = fh.readlines()
        if not header:  # first part's header, kept verbatim except the two recomputed fields
            header = [l for l in lines if not ANY.match(l.rstrip("\n")) and l.strip()]
        for l in lines:
            m = ANY.match(l.rstrip("\n"))
            if m:
                rows.append((int(m.group(2)), m.group(1)))

    rows.sort(key=lambda r: r[0])
    dup = len(rows) - len({r[0] for r in rows})
    if dup:
        print("[merge] REFUSING: %d duplicate seeds across parts" % dup, file=sys.stderr)
        return 1
    short = a.expect - len(rows)
    if short > 0:
        # Which pinned seeds produced nothing.  The intended list is the union of the
        # per-part slices, which is what the launcher actually handed the workers -- read
        # it from disk rather than trusting a count, so the repair knows names not totals.
        intended = []
        for part in a.parts:
            slice_path = os.path.join(a.stamp_dir, "_seeds_%s.txt" % part)
            if os.path.isfile(slice_path):
                with open(slice_path, encoding="utf-8") as fh:
                    intended += [int(s) for s in fh.read().split() if s.strip()]
        missing = sorted(set(intended) - {r[0] for r in rows})
        if not a.allow_short:
            print("[merge] REFUSING: got %d episodes, expected %d%s"
                  % (len(rows), a.expect,
                     ("; missing seeds " + " ".join(str(s) for s in missing)) if missing else ""),
                  file=sys.stderr)
            return 1
        if len(rows) < a.min_fraction * a.expect:
            print("[merge] REFUSING: got %d of %d episodes, below the %.0f%% floor"
                  % (len(rows), a.expect, 100 * a.min_fraction), file=sys.stderr)
            return 1
        with open(os.path.join(a.stamp_dir, "_missing_seeds.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(s) for s in missing) + ("\n" if missing else ""))
        print("[merge] SHORT: %d of %d episodes; %d missing seed(s) -> _missing_seeds.txt"
              % (len(rows), a.expect, len(missing)), file=sys.stderr)

    succ = sum(1 for _, text in rows if re.search(r"result:\s*Success", text))
    ran = len(rows)                     # the denominator is what ran, never what was asked for
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    instruction = "unseen"
    out = []
    for line in header:
        if line.startswith("Success Rate:"):
            out.append("Success Rate: %s (%d/%d)%s\n" % (
                succ / ran, succ, ran,
                "  [SHORT: %d seed(s) produced no episode]" % short if short > 0 else ""))
        elif line.startswith("End time:"):
            out.append("End time: %s\n" % now)
        else:
            if line.startswith("Instruction Type:"):
                instruction = line.split(":", 1)[1].strip()
            out.append(line if line.endswith("\n") else line + "\n")
    out.append("GPU parts: %d (%s)\n" % (len(a.parts), ", ".join(a.parts)))
    out.append("\n")
    for i, (_, text) in enumerate(rows):
        out.append("【%03d】%s\n" % (i, text))

    with open(os.path.join(a.stamp_dir, "_eval_log.txt"), "w", encoding="utf-8") as fh:
        fh.writelines(out)
    with open(os.path.join(a.stamp_dir, "_eval_seeds.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(s) for s, _ in rows) + "\n")
    with open(os.path.join(a.stamp_dir, "_result.txt"), "w", encoding="utf-8") as fh:
        fh.write("Timestamp: %s\n\n" % now)
        fh.write("Instruction Type: %s\n\n" % instruction)
        fh.write("%s\n" % (succ / a.expect))
    print("[merge] %d episodes, %d success (%.2f) -> %s"
          % (len(rows), succ, succ / a.expect, a.stamp_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
