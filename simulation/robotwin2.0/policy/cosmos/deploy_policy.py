"""RoboTwin 2.0 client adapter for the Cosmos3-Edge aloha-agilex action policy.

Client--server, the same split the pi0.5 adapter uses: cosmos-framework and
Sapien cannot share a venv, so RoboTwin only simulates and talks msgpack over a
websocket. The server is ``cosmos_robotwin.server`` in
``/DRoboticsAILab/vepfs_20260622/licheng.yang/cosmos-robotwin``.

    RoboTwin (.venv)                        cosmos-framework (.venv)
    policy/cosmos/deploy_policy.py  --ws-->  cosmos_robotwin.server
      3 views RGB (unflipped) + 14-D qpos     mosaic + flow-matching rollout
      take_action(qpos)                       16 x 14 absolute joint targets

What this adapter deliberately does NOT do:

* **No mosaic here.** The three views are sent raw (RGB on the wire, unflipped) and
  tiled server-side, next to the model, so the one train/serve-critical image
  op has a single owner.
* **No gripper flip and no normalisation.** RoboTwin's native convention (1.0
  open, 0.0 closed, raw joint units) is what the policy was trained on and what
  it returns, so the vector goes to ``take_action`` untouched.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Union

import numpy as np

from .ws_client import WebsocketClientPolicy

# Must match cosmos_robotwin.observation; duplicated (not imported) because the
# RoboTwin venv has no cosmos-framework on its path.
HEAD_IMAGE_KEY = "observation/head_image"
LEFT_WRIST_IMAGE_KEY = "observation/left_wrist_image"
RIGHT_WRIST_IMAGE_KEY = "observation/right_wrist_image"
JOINT_POSITION_KEY = "observation/joint_position"
PROMPT_KEY = "prompt"
NOISE_SEED_KEY = "noise_seed"
ACTION_RESPONSE_KEY = "actions"

ACTION_DIM = 14

_CAMERA_KEYS = {
    HEAD_IMAGE_KEY: "head_camera",
    LEFT_WRIST_IMAGE_KEY: "left_camera",
    RIGHT_WRIST_IMAGE_KEY: "right_camera",
}


def _as_rgb(hwc: np.ndarray) -> np.ndarray:
    """Validate a SAPIEN HWC uint8 frame and pass it through as **RGB**, unflipped.

    SAPIEN's live ``get_obs()`` really is RGB, so the wire carries RGB untouched.  This was
    ``_rgb_to_bgr``, flipping unconditionally for a BGR-expecting server; removed
    2026-09-12 when the eval clients were standardised on sending raw RGB.
    """
    arr = np.asarray(hwc, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb expected HWC uint8, got shape={arr.shape} dtype={arr.dtype}")
    return np.ascontiguousarray(arr)


def encode_obs(observation: dict) -> dict:
    """RoboTwin observation -> the websocket observation the server expects (RGB, unflipped)."""
    cameras = observation["observation"]
    out = {}
    for wire_key, camera in _CAMERA_KEYS.items():
        if camera not in cameras:
            raise KeyError(
                f"RoboTwin observation has no {camera!r}; the Cosmos policy needs all three of "
                f"{sorted(_CAMERA_KEYS.values())}. Check collect_wrist_camera in the task config."
            )
        out[wire_key] = _as_rgb(cameras[camera]["rgb"])

    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"expected a {ACTION_DIM}-D qpos vector, got shape {tuple(state.shape)}")
    out[JOINT_POSITION_KEY] = state
    return out


def _parse_replan_steps(raw: Union[int, str, None]) -> Optional[int]:
    """Fixed open-loop length, or None to execute the server's whole chunk."""
    if raw is None:
        return None
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("", "auto", "horizon", "none", "null"):
            return None
        return int(token)
    value = int(raw)
    return None if value <= 0 else value


class CosmosRemotePolicy:
    def __init__(self, usr_args: dict):
        self.task_name = usr_args["task_name"]
        # "env" uses RoboTwin's own instruction pool, which is what the finetune
        # trained on (the model card: instructions from the `seen` pool).
        self.prompt_mode = os.environ.get("COSMOS_PROMPT_MODE") or usr_args.get("prompt_mode", "env")
        self.replan_steps = _parse_replan_steps(usr_args.get("replan_steps", "auto"))
        self._logged_replan = False

        self.infer_latencies: list[float] = []
        self.last_infer_s: Optional[float] = None
        # Set per episode by eval_policy.py so a replay reuses the same sampler
        # noise; left None the server draws from its own RNG as usual.
        self.episode_seed: Optional[int] = None
        self.infer_index = 0

        host = usr_args.get("host") or os.environ.get("COSMOS_HOST", "127.0.0.1")
        port = int(usr_args.get("port") or os.environ.get("COSMOS_PORT", 8000))
        self.client = WebsocketClientPolicy(host=host, port=port)
        print(
            f"\033[36m[cosmos] server={host}:{port} prompt_mode={self.prompt_mode} "
            f"replan_steps={'auto' if self.replan_steps is None else self.replan_steps}\033[0m",
            flush=True,
        )

    # --- seeding -------------------------------------------------------------

    def set_episode_seed(self, seed) -> None:
        """Pin this episode's sampler noise; called once before step 0."""
        self.episode_seed = None if seed is None else int(seed)
        self.infer_index = 0

    def next_noise_seed(self) -> Optional[int]:
        """Seed for the next call, or None to let the server draw.

        ``episode_seed * 1_000_003 + infer_index``: a plain sum would alias
        (seed+1, call 0) onto (seed, call 1), so two different episodes would
        share a start latent. The large odd multiplier keeps the axes apart.
        """
        if self.episode_seed is None:
            return None
        seed = (self.episode_seed * 1_000_003 + self.infer_index) % (2**31)
        self.infer_index += 1
        return int(seed)

    # --- per-step helpers ----------------------------------------------------

    def resolve_prompt(self, TASK_ENV) -> str:
        return TASK_ENV.get_instruction()

    def resolve_exec_steps(self, actions: np.ndarray) -> int:
        horizon = int(actions.shape[0])
        if self.replan_steps is None:
            if not self._logged_replan:
                print(
                    f"\033[36m[cosmos] replan_steps=auto -> {horizon} (server action chunk)\033[0m",
                    flush=True,
                )
                self._logged_replan = True
            return horizon
        return min(self.replan_steps, horizon)

    def get_infer_stats(self) -> dict:
        """Per-episode websocket latency, in the keys eval_policy.py logs."""
        times = np.asarray(self.infer_latencies, dtype=np.float64)
        if times.size == 0:
            return {"infer_cnt_timed": 0, "infer_ms_mean": None, "infer_ms_max": None, "infer_time_s": 0.0}
        return {
            "infer_cnt_timed": int(times.size),
            "infer_ms_mean": float(times.mean() * 1000.0),
            "infer_ms_max": float(times.max() * 1000.0),
            "infer_time_s": float(times.sum()),
        }

    def reset(self) -> None:
        self.infer_latencies = []
        self.last_infer_s = None
        # Only the within-episode counter: eval_policy.py calls set_episode_seed
        # *before* reset_model, so clearing episode_seed here would drop the pin
        # that was just set.
        self.infer_index = 0


def get_model(usr_args: dict) -> CosmosRemotePolicy:
    return CosmosRemotePolicy(usr_args)


def eval(TASK_ENV, model: CosmosRemotePolicy, observation: dict) -> None:
    """Run one open-loop segment from a freshly predicted action chunk."""
    obs = encode_obs(observation)
    obs[PROMPT_KEY] = model.resolve_prompt(TASK_ENV)
    noise_seed = model.next_noise_seed()
    if noise_seed is not None:
        obs[NOISE_SEED_KEY] = noise_seed

    started = time.perf_counter()
    result = model.client.infer(obs)
    elapsed = time.perf_counter() - started
    model.last_infer_s = elapsed
    model.infer_latencies.append(elapsed)

    actions = np.asarray(result[ACTION_RESPONSE_KEY], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"server returned actions with shape {actions.shape}, expected [T,{ACTION_DIM}]")

    for action in actions[: model.resolve_exec_steps(actions)]:
        # Absolute joint targets in raw RoboTwin units, native gripper convention.
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model: CosmosRemotePolicy) -> None:
    model.reset()
