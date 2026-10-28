"""RoboTwin 2.0 client adapter for GR00T N1.7 (ZMQ / client-server)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Union

import numpy as np

from .zmq_client import ZmqPolicyClient, concat_joint_qpos

# Camera names produced by RoboTwin get_obs() -> GR00T video keys used in
# examples/RoboTwin/robotwin_aloha_config.py and checkpoint processor_config.
_CAMERAS = {
    "head_camera": "head",
    "left_camera": "left",
    "right_camera": "right",
}


def encode_obs(observation: dict) -> dict:
    """Map RoboTwin obs -> Gr00tPolicy nested video/state (batch=1, time=1).

    Channel order: the wire carries SAPIEN's **RGB** frames untouched.  Every weight in
    use is RGB-trained (the BGR->RGB flip lived only in the training front-end, where the
    LeRobot dump is BGR), so there is no flip here and none on the server
    (`ROBOTWIN_BGR2RGB` stays 0).  The client-side RGB->BGR flip that used to live here --
    correct only for the retired BGR-trained `joint_h32` weight -- was removed on
    2026-09-12 along with its `GR00T_CLIENT_RGB_TO_BGR` gate, so no environment variable
    can reintroduce it.
    """
    obs_cams = observation["observation"]
    vec = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
    if vec.shape[0] < 14:
        raise ValueError(f"joint_action.vector dim {vec.shape[0]} < 14")

    video = {}
    for src, dst in _CAMERAS.items():
        if src not in obs_cams:
            raise KeyError(f"RoboTwin obs missing camera '{src}' (need head/left/right)")
        hwc = np.asarray(obs_cams[src]["rgb"], dtype=np.uint8)
        if hwc.ndim != 3 or hwc.shape[-1] != 3:
            raise ValueError(f"rgb expected HWC uint8, got shape={hwc.shape} dtype={hwc.dtype}")
        hwc = np.ascontiguousarray(hwc)
        video[dst] = np.expand_dims(np.expand_dims(hwc, 0), 0)  # (1, 1, H, W, C)

    # Official 14-D: left_arm(6)+left_gripper(1)+right_arm(6)+right_gripper(1)
    state = {
        "left_arm": vec[0:6].reshape(1, 1, 6),
        "left_gripper": vec[6:7].reshape(1, 1, 1),
        "right_arm": vec[7:13].reshape(1, 1, 6),
        "right_gripper": vec[13:14].reshape(1, 1, 1),
    }
    return {"video": video, "state": state}


def _parse_replan_steps(raw: Union[int, str, None]) -> Optional[int]:
    """Return fixed replan length, or None to use the full action chunk."""
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "auto", "horizon", "none", "null"):
            return None
        return int(s)
    n = int(raw)
    return None if n <= 0 else n


class GR00TRemotePolicy:
    def __init__(self, usr_args: dict):
        self.task_name = usr_args["task_name"]
        self.prompt_mode = usr_args.get("prompt_mode") or os.environ.get(
            "GR00T_PROMPT_MODE", "env"
        )
        self.replan_steps = _parse_replan_steps(
            usr_args.get("replan_steps", os.environ.get("GR00T_REPLAN_STEPS", "auto"))
        )
        self._logged_replan = False
        self.infer_latencies = []
        self.last_infer_s = None
        self.episode_seed = None
        self.infer_index = 0
        self.chunk_records = []
        self._chunk_log_announced = False
        host = usr_args.get("host") or os.environ.get("GR00T_HOST", "127.0.0.1")
        port = int(usr_args.get("port") or os.environ.get("GR00T_PORT", 5555))
        timeout_ms = int(usr_args.get("timeout_ms") or os.environ.get("GR00T_TIMEOUT_MS", 600000))
        self.client = ZmqPolicyClient(host=host, port=port, timeout_ms=timeout_ms)
        if self.replan_steps is None:
            print(
                "\033[36m[gr00t] replan_steps=auto "
                "(will use action_horizon from first infer response)\033[0m",
                flush=True,
            )
        else:
            print(f"\033[36m[gr00t] replan_steps={self.replan_steps}\033[0m", flush=True)
        print(
            f"\033[36m[gr00t] prompt_mode={self.prompt_mode} server={host}:{port}\033[0m",
            flush=True,
        )
        print(
            f"\033[36m[gr00t] no-channel-flip (client sends SAPIEN RGB as-is) "
            f"ROBOTWIN_BGR2RGB={os.environ.get('ROBOTWIN_BGR2RGB', '0')}\033[0m",
            flush=True,
        )

    def set_episode_seed(self, seed) -> None:
        self.episode_seed = None if seed is None else int(seed)
        self.infer_index = 0
        self.chunk_records = []

    def next_noise_seed(self):
        """Deterministic inference-noise seed, mirroring the pi0.5 client.

        ``episode_seed * 1_000_003 + infer_index``: episodes differ by 1 and
        inference calls within an episode differ by 1, so a plain sum would
        alias -- (seed+1, call 0) and (seed, call 1) would share a seed and two
        different episodes would draw the same initial noise.  The prime spacing
        keeps every (episode, chunk) distinct while staying reproducible across
        runs.  Returns None when no episode seed is pinned, in which case the
        server falls back to its own unseeded draw (historical behaviour).
        """
        if self.episode_seed is None:
            return None
        return int((self.episode_seed * 1_000_003 + self.infer_index) % (2**31))

    def resolve_prompt(self, TASK_ENV) -> str:
        if self.prompt_mode == "train_short":
            return self.task_name.replace("_", " ")
        return TASK_ENV.get_instruction()

    def resolve_exec_steps(self, actions: np.ndarray) -> int:
        horizon = int(actions.shape[0])
        if self.replan_steps is None:
            if not self._logged_replan:
                print(
                    f"\033[36m[gr00t] replan_steps=auto -> {horizon} "
                    f"(action_horizon from server / weights)\033[0m",
                    flush=True,
                )
                self._logged_replan = True
            return horizon
        return min(self.replan_steps, horizon)

    def get_infer_stats(self) -> dict:
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
        self.infer_index = 0
        self.chunk_records = []
        self.client.reset()

    def record_chunk(
        self,
        actions: np.ndarray,
        *,
        exec_steps: int,
        infer_s: float,
        noise_seed,
        infer_index: int,
    ) -> None:
        """Keep one JSON-serialisable record per inference for the current episode."""
        arr = np.asarray(actions, dtype=np.float32)
        self.chunk_records.append(
            {
                "seed": self.episode_seed,
                "infer_index": int(infer_index),
                "noise_seed": None if noise_seed is None else int(noise_seed),
                "infer_s": round(float(infer_s), 6),
                "exec_steps": int(exec_steps),
                "horizon": int(arr.shape[0]),
                "action": np.round(arr, 6).tolist(),
            }
        )

    def flush_episode_chunks(self, path: str, extra: Optional[dict] = None) -> None:
        """Append this episode's inference chunks to ``path`` (JSONL)."""
        if not self.chunk_records:
            return
        extra = extra or {}
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for rec in self.chunk_records:
                row = dict(rec)
                for key, value in extra.items():
                    row.setdefault(key, value)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = len(self.chunk_records)
        self.chunk_records = []
        if not self._chunk_log_announced:
            print(f"\033[36m[gr00t] chunks jsonl -> {path}\033[0m", flush=True)
            self._chunk_log_announced = True
        print(f"\033[36m[gr00t] wrote {n} chunk(s) seed={self.episode_seed}\033[0m", flush=True)


def get_model(usr_args: dict) -> GR00TRemotePolicy:
    return GR00TRemotePolicy(usr_args)


def eval(TASK_ENV, model: GR00TRemotePolicy, observation: dict) -> None:
    """Run one open-loop segment from the latest GR00T action chunk.

    Server returns per-group arrays of length = train action_horizon (32).
    ``replan_steps=auto`` executes the full chunk; an int truncates before re-infer.
    """
    obs = encode_obs(observation)
    prompt = model.resolve_prompt(TASK_ENV)
    obs["language"] = {"annotation.human.task_description": [[prompt]]}
    # Pin the server's initial flow-matching noise to this (episode, chunk) when
    # the eval loop set an episode seed, so a replay of the same seedlist draws the
    # same noise call-for-call.  ``infer_index`` is read BEFORE the post-call bump,
    # so the first chunk of an episode uses index 0.  Sent only when a seed is
    # pinned; the server treats an absent key as "draw as usual", a no-op otherwise.
    noise_seed = model.next_noise_seed()
    options = {"noise_seed": noise_seed} if noise_seed is not None else None
    infer_idx = model.infer_index
    t0 = time.perf_counter()
    action_dict, _info = model.client.get_action(obs, options=options)
    infer_s = time.perf_counter() - t0
    model.last_infer_s = infer_s
    model.infer_latencies.append(infer_s)
    actions = concat_joint_qpos(action_dict)
    n = model.resolve_exec_steps(actions)
    model.record_chunk(
        actions,
        exec_steps=n,
        infer_s=infer_s,
        noise_seed=noise_seed,
        infer_index=infer_idx,
    )
    model.infer_index += 1
    for action in actions[:n]:
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model: GR00TRemotePolicy) -> None:
    model.reset()
