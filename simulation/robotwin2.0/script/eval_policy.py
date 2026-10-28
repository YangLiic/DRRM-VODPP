import sys
import os
import subprocess
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from description.utils.generate_episode_instructions import *


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
# os.chdir(os.path.join(os.getcwd(), "simulation/robotwin2.0"))


def format_result(key: int, res: dict):
    """Same per-episode line format as eval_policy_parallel.py."""
    s = f"【{key-1:03d}】"
    for k in [
        "seed",
        "success",
        "time",
        "fps",
        "action_cnt",
        "limit",
        "action_gen_freq",
        "infer_cnt",
        "infer_freq",
        "infer_ms_mean",
        "infer_ms_max",
        "infer_time_s",
        "start",
        "end",
        "count",
        "pid",
        "device",
    ]:
        if k not in res:
            continue
        elif k == "time":
            s += f"{k}: {int(res[k]):03d} s, "
        elif k == "infer_time_s":
            s += f"{k}: {float(res[k]):.3f} s, "
        elif k in ("infer_ms_mean", "infer_ms_max"):
            if res[k] is None:
                continue
            s += f"{k}: {float(res[k]):.1f} ms, "
        elif k in ["start", "end"]:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(res[k]))
            s += f"{k}: {timestamp}, "
        elif k == "fps" or "freq" in k:
            s += f"{k}: {res[k]:03.2f}, "
        elif k == "success":
            if res[k]:
                result = "Success"
            elif res[k] is None:
                result = "None   "
            else:
                result = "Fail   "
            s += f"result: {result}, "
        else:
            s += f"{k}: {res[k]}, "
    return s


def log_result(file_path, ind: int, result: dict, **kwargs):
    """Rewrite line ``ind`` in ``_eval_log.txt`` (single-process; no lock)."""
    result.update(**kwargs)
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        lines = f.readlines()
    if ind >= len(lines):
        lines += ["\n"] * (ind + 1 - len(lines))
    lines[ind] = format_result(ind, result) + "\n"
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        f.writelines(lines)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # Optional: EVAL_RESULT_ROOT / --eval_result_root (same layout as eval_policy_parallel.py)
    result_root = usr_args.get("eval_result_root")
    run_name = usr_args.get("eval_run_name") or ckpt_setting
    # EVAL_RESULT_CONFIG only relocates the result tree (e.g. clean_bgr). The
    # loaded task_config yml is unchanged, so scenes stay demo_clean.
    result_config = (
        usr_args.get("eval_result_config")
        or os.environ.get("EVAL_RESULT_CONFIG")
        or task_config
    )
    if result_root:
        save_dir = (
            Path(result_root) / task_name / policy_name / result_config / str(run_name) / current_time
        )
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{result_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Always keep a save dir for _eval_log.txt (parallel uses eval_video_save_dir).
    video_save_dir = save_dir
    args["eval_video_save_dir"] = str(video_save_dir)
    if args.get("eval_video_log"):
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
    else:
        video_size = None

    # Keep schedule / run metadata for the final log header (parallel-compatible).
    args["fm_tau_schedule"] = usr_args.get(
        "fm_tau_schedule",
        usr_args.get(
            "inference_tau_schedule",
            os.environ.get("PI05_FM_TAU_SCHEDULE") or os.environ.get("GR00T_TAU_NODES"),
        ),
    )
    args["eval_run_name"] = run_name
    args["eval_result_root"] = result_root
    args["eval_result_config"] = result_config
    args["replan_steps"] = usr_args.get("replan_steps")

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = 1

    # Validate the pinned seed list before paying for the checkpoint load: a short list
    # is a configuration error, and finding it after start-up wastes the whole arm.
    load_pinned_seeds(test_num)

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def load_pinned_seeds(test_num):
    """Read ``ROBOTWIN_SEED_LIST`` and refuse a short list -- BEFORE anything expensive.

    This is called twice on purpose: once in ``main`` before the policy is loaded, and
    once inside ``eval_policy`` where the list is actually used.  The early call is what
    makes a short list cheap: the check used to live only in ``eval_policy``, which runs
    after ``get_model`` has loaded the checkpoint and (for GR00T/pi0.5) spun up the
    serving processes, so a one-seed shortfall cost a full arm's start-up before failing
    -- measured 2026-08-30 on open_laptop, whose list holds 99 while the arm asked 100.
    """
    raw = os.environ.get("ROBOTWIN_SEED_LIST", "").strip()
    if not raw:
        return None
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as handle:
            raw = handle.read()
    seeds = [int(s) for s in raw.replace("\n", ",").split(",") if s.strip()]
    if len(seeds) < test_num:
        raise ValueError(
            f"ROBOTWIN_SEED_LIST has {len(seeds)} seeds but test_num={test_num}; "
            "a short list would silently shorten the run.  Either lower test_num to "
            f"{len(seeds)} or top the list up (exsched tools/topup_arms.py)."
        )
    return seeds


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    # ------------------------------------------------------------------ seeds
    # By default the episodes are *discovered*: walk forward from st_seed and
    # keep the seeds the expert planner can solve.  That acceptance test is
    # stochastic (the planner is randomised), so two runs of the same task drift
    # apart -- once one seed is judged differently, every later seed differs.
    # Measured on scan_object/demo_clean: uniform5 started 100006, 100011, ...
    # while uniform3 started 100002, 100006, ...  Comparing success rates across
    # runs then compares *different scenes*, which is not a comparison at all.
    #
    # ROBOTWIN_SEED_LIST pins the episodes instead: every run -- every schedule,
    # every difficulty -- evaluates exactly these seeds, in this order, so the
    # comparison is paired.  Unset keeps the historical discover-as-you-go path.
    #
    # The expert check still runs (line ~371 needs its episode_info for the
    # language instruction), but it can no longer change *which* seed we are on:
    # a failure retries the same seed instead of stepping past it.
    fixed_seeds = load_pinned_seeds(test_num)
    if fixed_seeds:
        print(
            f"\033[36m[seed-list] pinned to {len(fixed_seeds)} seeds "
            f"({fixed_seeds[0]}..{fixed_seeds[-1]}), expert check no longer selects\033[0m",
            flush=True,
        )
    fixed_idx = 0
    seed_attempts = 0
    SEED_MAX_ATTEMPTS = 3

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # Parallel-compatible episode log (updated live; finalized at end).
    log_dir = args.get("eval_video_save_dir")
    if log_dir is None:
        raise RuntimeError("eval_video_save_dir is required for _eval_log.txt")
    log_path = os.path.join(log_dir, "_eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as file:
        file.write(
            f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
            + "\n" * test_num
        )
    print(f"\033[36m[eval] writing live log -> {log_path}\033[0m", flush=True)

    # When video is disabled, do not pass save dir into env (avoids ffmpeg writes).
    save_video = bool(args.get("eval_video_log", False)) and video_size is not None
    if not save_video:
        args = dict(args)
        args["eval_video_save_dir"] = None

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        gpu_id_int = int(gpu_id)
    except ValueError:
        gpu_id_int = gpu_id

    def _seed_rejected(reason):
        """Advance past a seed the expert could not set up.

        Discovery mode steps to the next seed.  Pinned mode must not: stepping
        would silently change the episode set, which is exactly what this is
        here to prevent.  It retries the same seed (the planner is stochastic,
        and these seeds were validated when the list was produced) and only
        gives up -- loudly -- after SEED_MAX_ATTEMPTS.
        """
        nonlocal now_seed, fixed_idx, seed_attempts
        if fixed_seeds is None:
            now_seed += 1
            return
        seed_attempts += 1
        if seed_attempts >= SEED_MAX_ATTEMPTS:
            print(
                f"\033[31m[seed-list] seed={now_seed} unusable after "
                f"{seed_attempts} tries ({reason}); dropping it. "
                f"THIS RUN IS NO LONGER PAIRED WITH THE OTHERS.\033[0m",
                flush=True,
            )
            fixed_idx += 1
            seed_attempts = 0

    while succ_seed < test_num:
        if fixed_seeds is not None:
            if fixed_idx >= len(fixed_seeds):
                print(f"\033[31m[seed-list] exhausted at {succ_seed}/{test_num}\033[0m", flush=True)
                break
            now_seed = fixed_seeds[fixed_idx]

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(f"skip unstable seed={now_seed}: {e}", flush=True)
                TASK_ENV.close_env()
                _seed_rejected("unstable")
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(f"expert_check error seed={now_seed}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                TASK_ENV.close_env()
                _seed_rejected(type(e).__name__)
                args["render_freq"] = render_freq
                continue

        # In pinned mode the expert's verdict is recorded but never used to
        # select: the seed set is fixed, so a scene the expert fails today is
        # still evaluated (and would be evaluated by every other run too).
        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
            if fixed_seeds is not None:
                fixed_idx += 1
                seed_attempts = 0
        elif fixed_seeds is not None:
            print(f"[seed-list] expert failed on pinned seed={now_seed}, evaluating anyway", flush=True)
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
            fixed_idx += 1
            seed_attempts = 0
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        log_ind = now_id + 1
        log_info = {"seed": now_seed, "success": None}
        eval_start = time.time()
        log_result(
            log_path,
            log_ind,
            log_info,
            start=eval_start,
            device=gpu_id_int,
            pid=os.getpid(),
        )

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)

        # ``now_seed`` used to reach the scene and nothing else, so two runs of
        # the same schedule on the same pinned seed list still differed in two
        # places: which unseen instruction was drawn below, and the policy's
        # inference noise.  Both are re-derived from the seed here, which is
        # what makes a replay comparable episode by episode.
        np.random.seed(now_seed)
        if hasattr(model, "set_episode_seed"):  # opt-in; other policies untouched
            model.set_episode_seed(now_seed)

        episode_info_list = [episode_info["info"]]
        # ``test_num`` is this PROCESS's episode count, but here it is the size of
        # the instruction pool -- and ``np.random.choice`` over a 25-entry pool
        # picks a different sentence than over a 100-entry one for the same seed.
        # Splitting a 100-episode arm across 4 workers therefore silently changed
        # the policy's prompt, making the run incomparable with every serial arm
        # ever recorded.  ``ROBOTWIN_DESC_NUM`` decouples the two; unset it keeps
        # the historical behaviour exactly.
        desc_num = int(os.environ.get("ROBOTWIN_DESC_NUM") or test_num)
        results = generate_episode_descriptions(args["task_name"], episode_info_list, desc_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        infer_cnt = 0
        limit = TASK_ENV.step_lim
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            infer_cnt += 1
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        eval_end = time.time()
        eval_time = eval_end - eval_start
        action_cnt = TASK_ENV.take_action_cnt
        infer_stats = {}
        if hasattr(model, "get_infer_stats"):
            infer_stats = model.get_infer_stats()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        if infer_stats.get("infer_ms_mean") is not None:
            print(
                f"\033[K[Infer] mean {infer_stats['infer_ms_mean']:.1f} ms | "
                f"max {infer_stats['infer_ms_max']:.1f} ms | "
                f"total {infer_stats['infer_time_s']:.3f} s | "
                f"calls {infer_stats['infer_cnt_timed']}"
            )

        log_result(
            log_path,
            log_ind,
            log_info,
            success=succ,
            end=eval_end,
            time=eval_time,
            fps=render_freq,
            action_cnt=action_cnt,
            limit=limit,
            action_gen_freq=(action_cnt / eval_time) if eval_time > 0 else 0.0,
            infer_cnt=infer_cnt,
            infer_freq=(infer_cnt / eval_time) if eval_time > 0 else 0.0,
            infer_ms_mean=infer_stats.get("infer_ms_mean"),
            infer_ms_max=infer_stats.get("infer_ms_max"),
            infer_time_s=infer_stats.get("infer_time_s", 0.0),
        )
        if hasattr(model, "flush_episode_chunks"):
            model.flush_episode_chunks(
                os.path.join(log_dir, "_chunks.jsonl"),
                extra={
                    "task": task_name,
                    "run": args.get("eval_run_name"),
                    "success": bool(succ),
                    "action_cnt": int(action_cnt),
                    "infer_cnt": int(infer_cnt),
                    "tau_nodes": args.get("fm_tau_schedule"),
                },
            )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    # The seeds this run actually evaluated, one per line, next to the log.
    # In discovery mode this is the artifact to promote into a canonical list
    # (feed it back via ROBOTWIN_SEED_LIST so later runs are paired with it);
    # in pinned mode it is the receipt that the pinning held.
    seed_path = os.path.join(log_dir, "_eval_seeds.txt")
    with open(seed_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(str(s) for s in suc_test_seed_list) + "\n")
    print(f"\033[36m[eval] seeds -> {seed_path}\033[0m", flush=True)

    # Finalize header (same layout as eval_policy_parallel.py).
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        lines = f.readlines()
    summary = [
        f"Task Name: {task_name}\n",
        f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n",
        f"Instruction Type: {instruction_type}\n",
        f"Success Rate: {TASK_ENV.suc / test_num} ({TASK_ENV.suc}/{test_num})\n",
    ]
    if args.get("eval_run_name"):
        summary.append(f"Eval Run Name: {args['eval_run_name']}\n")
    if args.get("fm_tau_schedule") is not None:
        summary.append(f"FM tau schedule: {args['fm_tau_schedule']}\n")
    if args.get("replan_steps") is not None:
        summary.append(f"replan_steps: {args['replan_steps']}\n")
    if args.get("ckpt_setting"):
        summary.append(f"Checkpoint: {args['ckpt_setting']}\n")
    summary.append("\n")
    if len(lines) > 0:
        lines = summary[0:1] + lines[0:1] + summary[1:] + lines[1:]
    else:
        lines = summary
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Data has been saved to {log_path}")

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
