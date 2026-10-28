#!/usr/bin/env python3
"""ZMQ/msgpack inference server for VODPPlus checkpoints.

This server is compatible with TeleManipulation's PolicyClient:
request  = msgpack({"endpoint": "get_action", "data": obs})
response = msgpack({"actions": np.ndarray})
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
import zmq

from drrm.models.base_runner import load_policy


DEFAULT_IMAGE_KEY_MAP = {
    "observation.images.hand_left": "left_cam",
    "observation.images.hand_right": "right_cam",
    "observation.images.top_head": "head_cam",
}
DEFAULT_ACTION_STATE_INDICES = [
    0, 1, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
]


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {key: _encode_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if b"__ndarray__" in value or "__ndarray__" in value:
            data = value.get(b"data", value.get("data"))
            dtype = value.get(b"dtype", value.get("dtype"))
            shape = value.get(b"shape", value.get("shape"))
            if isinstance(dtype, bytes):
                dtype = dtype.decode()
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {
            (key.decode() if isinstance(key, bytes) else key): _decode_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _pack(data: dict[str, Any]) -> bytes:
    return msgpack.packb(_encode_value(data), use_bin_type=True)


def _unpack(data: bytes) -> dict[str, Any]:
    return _decode_value(msgpack.unpackb(data, raw=False))


def _image_to_tensor(
    image: Any,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: tuple[int, int] | None,
) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 3:
        # GenericPolicy decodes with OpenCV, so ndarray images arrive as BGR HWC.
        array = array[..., ::-1].copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    elif array.ndim == 3 and array.shape[0] == 3:
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(f"Expected image with shape HWC or CHW, got {array.shape}")

    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.numel() > 0 and tensor.max() > 2:
        tensor = tensor / 255.0
    if resize_hw is not None and tuple(tensor.shape[-2:]) != resize_hw:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=resize_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.unsqueeze(0).unsqueeze(0)


def _state_to_tensor(state: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(state, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected state shape [D], [T,D], or [B,T,D], got {tuple(tensor.shape)}")
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


class VODPPServer:
    def __init__(
        self,
        checkpoint_dir: str,
        device: str,
        dtype: str,
        image_key_map: dict[str, str],
        action_state_indices: list[int],
    ):
        self.device = torch.device(device)
        self.policy = load_policy(checkpoint_dir, use_ckp_code=False, device=device)
        if dtype == "fp32":
            torch_dtype = torch.float32
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "auto":
            torch_dtype = None
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if torch_dtype is None:
            self.policy.to(self.device)
        else:
            self.policy.to(self.device, dtype=torch_dtype)
        self.policy.eval()
        if hasattr(self.policy, "normalizer"):
            self.policy.normalizer.to(self.device)

        self.image_key_map = image_key_map
        self.action_state_indices = action_state_indices
        self.dtype = self.policy.dtype
        self.image_resize_hw = self._get_image_resize_hw()

    def _get_image_resize_hw(self) -> dict[str, tuple[int, int]]:
        shape_meta = self.policy.config.shape_meta["obs"]
        result = {}
        for target_key in self.image_key_map.values():
            shape = shape_meta[target_key]["shape"]
            result[target_key] = (int(shape[-2]), int(shape[-1]))
        return result

    def get_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        model_obs: dict[str, torch.Tensor] = {}

        for source_key, target_key in self.image_key_map.items():
            if source_key in obs:
                model_obs[target_key] = _image_to_tensor(
                    obs[source_key],
                    self.device,
                    self.dtype,
                    self.image_resize_hw.get(target_key),
                )
            elif target_key in obs:
                model_obs[target_key] = _image_to_tensor(
                    obs[target_key],
                    self.device,
                    self.dtype,
                    self.image_resize_hw.get(target_key),
                )

        if "state" in obs:
            state = obs["state"]
        elif "action" in obs:
            state = np.asarray(obs["action"], dtype=np.float32)[self.action_state_indices]
        elif "observation.state" in obs:
            raw_state = np.asarray(obs["observation.state"], dtype=np.float32)
            if raw_state.shape[-1] != len(self.action_state_indices):
                raise ValueError(
                    "Request has observation.state but no 20D state/action field. "
                    f"Got observation.state dim={raw_state.shape[-1]}."
                )
            state = raw_state
        else:
            raise KeyError("Request must contain 'state' or 'action'.")

        model_obs["state"] = _state_to_tensor(state, self.device, self.dtype)

        missing = [key for key in ("left_cam", "right_cam", "head_cam", "state") if key not in model_obs]
        if missing:
            raise KeyError(f"Missing VODPP inputs: {missing}")

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = self.policy.predict_action(model_obs)
            else:
                result = self.policy.predict_action(model_obs)
        actions = result["action"].detach().cpu().float().numpy()
        return {"actions": actions}

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "model": "vodpp",
            "action_dim": self.policy.action_dim,
            "horizon": self.policy.horizon,
            "n_action_steps": self.policy.n_action_steps,
            "n_obs_steps": self.policy.n_obs_steps,
            "image_key_map": self.image_key_map,
            "action_state_indices": self.action_state_indices,
        }

    def reset(self, _options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True}

    def ping(self) -> dict[str, Any]:
        return {"ok": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="/DRoboticsAILab/users/licheng.yang/checkpoints/pick_block_color_convert")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5650)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16", "auto"], default="fp32")
    parser.add_argument("--image-key-map", type=json.loads, default=DEFAULT_IMAGE_KEY_MAP)
    parser.add_argument("--action-state-indices", type=json.loads, default=DEFAULT_ACTION_STATE_INDICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    server = VODPPServer(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        image_key_map=args.image_key_map,
        action_state_indices=[int(idx) for idx in args.action_state_indices],
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"Server is ready and listening on tcp://{args.host}:{args.port}")
    print(f"Checkpoint: {checkpoint_dir}")

    try:
        while True:
            request = _unpack(socket.recv())
            endpoint = request.get("endpoint")
            data = request.get("data", {})
            try:
                method = getattr(server, endpoint)
                if endpoint in {"get_action", "reset"}:
                    response = method(data)
                else:
                    response = method()
            except Exception as exc:
                response = {"error": str(exc)}
            socket.send(_pack(response))
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
