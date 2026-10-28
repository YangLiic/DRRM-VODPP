"""RoboTwin 2.0 client adapter for OpenPI pi0.5 (websocket / Scheme A)."""

from __future__ import annotations

import os
import time
from typing import Optional, Union

import numpy as np

from .ws_client import WebsocketClientPolicy

# Matches LeRobot meta/tasks.jsonl used in pi05_robotwin2.0_agilex training.
_TRAIN_SHORT_PROMPTS = {
    "adjust_bottle": "adjust bottle",
    "scan_object": "scan object",
    "beat_block_hammer": "beat block hammer",
}


# RoboTwin always hands over the same observation; which vocabulary the policy
# wants depends on the openpi data pipeline its weights were trained with.
#
#   agilex -- RobotWinAgilexInputs: flat head_cam/left_cam/right_cam, uint8 HWC.
#             Used by pi05_robotwin2.0_agilex.
#   aloha  -- AlohaInputs: nested images{cam_high,cam_left_wrist,cam_right_wrist}
#             and uint8 **CHW** -- it runs rearrange("c h w -> h w c")
#             unconditionally, and rejects camera names outside
#             EXPECTED_CAMERAS.  Used by pi05_motus_robotwin, whose weights come
#             from the LeRobotAlohaDataConfig pipeline.
#
# Emitting the right shape here is what keeps openpi untouched:
# serve_policy.py leaves repack_transforms empty, so whatever this sends goes
# straight into the policy's own input transform.  Renaming on the server side
# instead would need the *training* repack, whose source keys are the training
# data's column names, not the ones a live robot sends.
_ALOHA_CAMERAS = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _as_rgb(hwc: np.ndarray) -> np.ndarray:
    """Validate a SAPIEN HWC uint8 frame and pass it through as **RGB**, unflipped.

    The pi0.5 weights in use are RGB-trained (the BGR->RGB flip lived only in the training
    front-end, where the LeRobot dump is BGR), so the wire carries RGB and nothing flips
    here or on the server.  This used to be `_rgb_to_bgr`, which flipped unconditionally --
    correct only for the retired BGR-trained weights; removed 2026-09-12.
    """
    arr = np.asarray(hwc, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb expected HWC uint8, got shape={arr.shape} dtype={arr.dtype}")
    return np.ascontiguousarray(arr)


def encode_obs(observation: dict, obs_format: str = "agilex") -> dict:
    """Map RoboTwin obs -> the key vocabulary the configured policy expects.

    Frames go out as SAPIEN's own RGB, untouched; see ``_as_rgb``.
    """
    obs_cams = observation["observation"]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

    if obs_format == "aloha":
        images = {}
        for camera, name in _ALOHA_CAMERAS.items():
            if camera not in obs_cams:
                continue
            hwc = _as_rgb(obs_cams[camera]["rgb"])
            images[name] = np.transpose(hwc, (2, 0, 1))
        return {"images": images, "state": state}

    if obs_format != "agilex":
        raise ValueError(f"unknown obs_format {obs_format!r}; want 'agilex' or 'aloha'")

    out = {
        "head_cam": _as_rgb(obs_cams["head_camera"]["rgb"]),
        "state": state,
    }
    if "left_camera" in obs_cams:
        out["left_cam"] = _as_rgb(obs_cams["left_camera"]["rgb"])
    if "right_camera" in obs_cams:
        out["right_cam"] = _as_rgb(obs_cams["right_camera"]["rgb"])
    return out


def _parse_replan_steps(raw: Union[int, str, None]) -> Optional[int]:
    """Return fixed replan length, or None to use server action_horizon.

    Accepted auto tokens: auto / horizon / none / null / empty / <=0.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "auto", "horizon", "none", "null"):
            return None
        return int(s)
    n = int(raw)
    return None if n <= 0 else n


class PI05RemotePolicy:
    def __init__(self, usr_args: dict):
        self.task_name = usr_args["task_name"]
        self.prompt_mode = usr_args.get("prompt_mode", "train_short")
        # Which key vocabulary this checkpoint's input transform expects.
        # Env var wins so a sweep can flip it per job without editing the yml.
        self.obs_format = os.environ.get("PI05_OBS_FORMAT") or usr_args.get(
            "obs_format", "agilex"
        )
        # None => execute full action chunk (= model action_horizon from weights/config).
        self.replan_steps = _parse_replan_steps(usr_args.get("replan_steps", "auto"))
        self._logged_replan = False
        self.infer_latencies = []
        self.last_infer_s = None
        # Deterministic inference noise.  ``episode_seed`` is set per episode by
        # the eval loop; ``infer_index`` counts inference calls *within* that
        # episode, so call k of a replay lands on the same start latent as call
        # k of the original run.  Left None (nobody set a seed) no ``noise_seed``
        # key is ever sent and the server draws exactly as it always has.
        self.episode_seed = None
        self.infer_index = 0
        host = usr_args.get("host") or os.environ.get("PI05_HOST", "127.0.0.1")
        port = int(usr_args.get("port") or os.environ.get("PI05_PORT", 8000))
        self.client = WebsocketClientPolicy(host=host, port=port)
        if self.replan_steps is None:
            print(
                "\033[36m[pi05] replan_steps=auto "
                "(will use action_horizon from first infer response)\033[0m",
                flush=True,
            )
        else:
            print(f"\033[36m[pi05] replan_steps={self.replan_steps}\033[0m", flush=True)

    def set_episode_seed(self, seed) -> None:
        """Pin this episode's inference noise; call once per episode, before step 0."""
        self.episode_seed = None if seed is None else int(seed)
        self.infer_index = 0

    def next_noise_seed(self):
        """Seed for the *next* inference call, or None to leave the server alone.

        ``episode_seed * 1_000_003 + infer_index``: episodes differ by 1 and
        calls differ by 1, so a plain sum would alias -- (seed+1, call 0) and
        (seed, call 1) would collide and two different episodes would share a
        start latent.  A large odd multiplier separates the two axes.
        """
        if self.episode_seed is None:
            return None
        seed = (self.episode_seed * 1_000_003 + self.infer_index) % (2**31)
        self.infer_index += 1
        return int(seed)

    def resolve_prompt(self, TASK_ENV) -> str:
        if self.prompt_mode == "env":
            return TASK_ENV.get_instruction()
        return _TRAIN_SHORT_PROMPTS.get(self.task_name, TASK_ENV.get_instruction())

    def resolve_exec_steps(self, actions: np.ndarray) -> int:
        horizon = int(actions.shape[0])
        if self.replan_steps is None:
            if not self._logged_replan:
                print(
                    f"\033[36m[pi05] replan_steps=auto -> {horizon} "
                    f"(action_horizon from server / weights)\033[0m",
                    flush=True,
                )
                self._logged_replan = True
            return horizon
        return min(self.replan_steps, horizon)

    def get_infer_stats(self) -> dict:
        """Per-episode websocket infer latency (same keys as VODPPRunner)."""
        times = list(self.infer_latencies)
        if not times:
            return {
                "infer_cnt_timed": 0,
                "infer_ms_mean": None,
                "infer_ms_max": None,
                "infer_time_s": 0.0,
            }
        arr = np.asarray(times, dtype=np.float64)
        return {
            "infer_cnt_timed": int(arr.size),
            "infer_ms_mean": float(arr.mean() * 1000.0),
            "infer_ms_max": float(arr.max() * 1000.0),
            "infer_time_s": float(arr.sum()),
        }

    def reset(self) -> None:
        self.infer_latencies = []
        self.last_infer_s = None
        # Only the within-episode call counter.  ``reset_model`` runs *after*
        # the eval loop has already called ``set_episode_seed`` for this
        # episode (eval_policy.py: seed at ~458, reset at ~497), so clearing
        # ``episode_seed`` here would silently drop the pin that was just set.
        self.infer_index = 0


def get_model(usr_args: dict) -> PI05RemotePolicy:
    return PI05RemotePolicy(usr_args)


def eval(TASK_ENV, model: PI05RemotePolicy, observation: dict) -> None:
    """Run one open-loop segment from the latest action chunk.

    Server returns ``actions`` with length = train ``action_horizon`` (e.g. 32).
    ``replan_steps=auto`` executes the full chunk; an int truncates before re-infer.

    Eval video frames are written inside ``TASK_ENV.take_action`` after each
    executed step (fresh head-camera capture).
    """
    obs = encode_obs(observation, model.obs_format)
    obs["prompt"] = model.resolve_prompt(TASK_ENV)
    # Only sent when an episode seed was pinned; the server treats an absent key
    # as "draw as usual", so this is a no-op for unseeded runs.
    noise_seed = model.next_noise_seed()
    if noise_seed is not None:
        obs["noise_seed"] = noise_seed
    t0 = time.perf_counter()
    result = model.client.infer(obs)
    infer_s = time.perf_counter() - t0
    model.last_infer_s = infer_s
    model.infer_latencies.append(infer_s)
    actions = np.asarray(result["actions"])
    n = model.resolve_exec_steps(actions)
    for action in actions[:n]:
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model: PI05RemotePolicy) -> None:
    model.reset()
