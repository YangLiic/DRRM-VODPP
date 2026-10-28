"""Mine the eval seed list with the expert planner only -- no policy, no video.

Seed validity in ``eval_policy.py`` discovery mode is decided *before* the
policy ever runs: a seed is kept iff the curobo-based expert
(``envs/robot/planner.py``) can solve the scene --

    TASK_ENV.setup_demo(seed=s, is_test=True, **args)
    TASK_ENV.play_once()
    keep = TASK_ENV.plan_success and TASK_ENV.check_success()

so discovering seeds by running the full closed-loop eval (pi0.5 load, VLM
forward per chunk, ffmpeg) pays ~2.5h for a decision the expert makes alone in
minutes.  This script walks the identical predicate over the identical seed
sequence (st_seed = 100000*(1+seed), consecutive) and writes the accepted
seeds to ``--out`` -- byte-compatible with ``_eval_seeds.txt`` /
``ROBOTWIN_SEED_LIST``.

Two properties keep this faithful to what a discovery eval would have pinned:

  * same predicate, same task_config (``is_test=True`` included -- collect_data
    omits it, which is why that script's seed.txt is NOT a substitute);
  * the planner is stochastic, so this list is *a* valid draw rather than
    *the* canonical truth -- exactly like a discovery run.  Pinned replays
    evaluate expert-failed seeds anyway (``eval_policy.py``), so downstream
    runs never deadlock on planner noise.

Usage (from simulation/robotwin2.0, cloud render env sourced):

    $PYTHON script/mine_eval_seeds.py --task_name blocks_ranking_rgb \
        --task_config demo_clean --test_num 100 --out /path/seedlist.txt
"""

import sys
import os
import argparse
import time

sys.path.append("./")
sys.path.append("./description/utils")

import yaml

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

# Reuse eval_policy's own helpers so env construction cannot drift from the
# real eval path.  Importing eval_policy is safe: model loading only happens
# inside main().
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_policy import class_decorator, get_embodiment_config  # noqa: E402


def build_args(task_name, task_config):
    """The env-relevant half of eval_policy.main(): task yml + embodiment."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = _embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Expert-only: no rendering to screen, no eval video.
    args["render_freq"] = 0
    args["eval_video_save_dir"] = None
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--test_num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="st_seed = 100000*(1+seed)")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max_walk",
        type=int,
        default=2000,
        help="give up after this many candidate seeds (asset gaps can make a "
        "config unmineable; fail loudly rather than walk forever)",
    )
    ap.add_argument(
        "--topup",
        action="store_true",
        help="keep the seeds already in --out and walk on from where they stop, "
        "until test_num is reached.  Without this a short list can only be "
        "replaced, never completed: the walk restarts at st_seed, re-pays for "
        "every seed it already knows, and rewrites --out from scratch.  That is "
        "how three lists on this stack ended up pinned below 100 (open_microwave "
        "92, grab_roller/open_laptop 99) after an early max_walk=2000 run -- the "
        "shortfall then travelled silently into every campaign that read them, "
        "because the arm scripts take the seed count from the file.",
    )
    a = ap.parse_args()

    args = build_args(a.task_name, a.task_config)
    TASK_ENV = class_decorator(a.task_name)

    st_seed = 100000 * (1 + a.seed)
    accepted = []
    rejects = {}
    started = time.time()

    # Top-up resumes AFTER the highest seed already accepted.  The walk is over a
    # consecutive sequence and the predicate is deterministic per seed up to planner
    # noise, so everything in [st_seed, max(existing)] has already been judged: re-walking
    # it would buy nothing and cost the whole original run again.  Resuming past the end
    # keeps the existing prefix byte-identical, which is what makes a topped-up list still
    # pair seed-for-seed with every arm already run against the short one.
    resume_from = st_seed
    if a.topup and os.path.isfile(a.out):
        with open(a.out, encoding="utf-8") as handle:
            accepted = [int(v) for v in handle.read().split()]
        if len(accepted) >= a.test_num:
            print(f"[mine] {a.out} already holds {len(accepted)} >= {a.test_num} seeds; "
                  "nothing to top up", flush=True)
            sys.exit(0)
        resume_from = max(accepted) + 1 if accepted else st_seed
        print(f"[mine] top-up: keeping {len(accepted)} seed(s), resuming at "
              f"{resume_from} (walked {resume_from - st_seed} previously), "
              f"need {a.test_num - len(accepted)} more", flush=True)

    now_seed = resume_from
    # ``max_walk`` bounds THIS run's walk, not the distance from the original start:
    # a top-up gets its own budget, otherwise resuming past an exhausted one would exit
    # immediately and look like a task that cannot be mined.
    while len(accepted) < a.test_num and (now_seed - resume_from) < a.max_walk:
        try:
            TASK_ENV.setup_demo(now_ep_num=len(accepted), seed=now_seed, is_test=True, **args)
            TASK_ENV.play_once()
            TASK_ENV.close_env()
        except UnStableError as e:
            print(f"[mine] seed={now_seed} unstable: {e}", flush=True)
            TASK_ENV.close_env()
            rejects["unstable"] = rejects.get("unstable", 0) + 1
            now_seed += 1
            continue
        except Exception as e:
            print(f"[mine] seed={now_seed} expert error: {type(e).__name__}: {e}", flush=True)
            TASK_ENV.close_env()
            rejects[type(e).__name__] = rejects.get(type(e).__name__, 0) + 1
            now_seed += 1
            continue

        if TASK_ENV.plan_success and TASK_ENV.check_success():
            accepted.append(now_seed)
            if len(accepted) % 10 == 0:
                rate = (time.time() - started) / max(1, now_seed - st_seed)
                print(
                    f"[mine] {len(accepted)}/{a.test_num} accepted "
                    f"(walked {now_seed - st_seed + 1}, {rate:.1f}s/seed)",
                    flush=True,
                )
        else:
            rejects["plan_or_check_fail"] = rejects.get("plan_or_check_fail", 0) + 1
        now_seed += 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(str(s) for s in accepted) + "\n")

    ok = len(accepted) >= a.test_num
    print(
        f"[mine] {'DONE' if ok else 'INCOMPLETE'}: {len(accepted)}/{a.test_num} seeds "
        f"in {time.time() - started:.0f}s, walked {now_seed - resume_from} this run, "
        f"rejects={rejects} -> {a.out}",
        flush=True,
    )
    if not ok:
        # Say what to do about it, here, where the shortfall is first known.  A list that
        # stops short does not announce itself downstream -- the arm scripts read the file
        # and evaluate whatever length they find -- so the remedy belongs in this message.
        print(
            f"[mine] SHORTFALL {a.test_num - len(accepted)} seed(s).  Re-run with "
            f"--topup and a larger --max_walk to keep these {len(accepted)} and continue "
            f"past seed {now_seed}; every arm already run on the short list stays "
            "seed-for-seed comparable, because top-up only appends.",
            flush=True,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
