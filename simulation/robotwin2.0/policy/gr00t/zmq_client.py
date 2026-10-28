"""Minimal GR00T ZMQ client (msgpack-numpy). No Isaac-GR00T import required.

Wire protocol matches ``gr00t.policy.server_client.PolicyServer``:
request  = msgpack({"endpoint": "...", "data": {...}})
response = msgpack(result)  # get_action -> [action_dict, info_dict]
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=mnp.encode, use_bin_type=True)


def _from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


class ZmqPolicyClient:
    """Talk to ``gr00t.eval.run_gr00t_server`` over ZMQ REQ/REP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 600_000,
        connect_timeout_s: int = 600,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self._init_socket()
        self._wait_until_ready(connect_timeout_s)

    def _init_socket(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _wait_until_ready(self, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        logging.info("Waiting for GR00T server at %s:%s ...", self.host, self.port)
        while True:
            try:
                self.socket.setsockopt(zmq.RCVTIMEO, 2000)
                self.socket.setsockopt(zmq.SNDTIMEO, 2000)
                self.call_endpoint("ping", requires_input=False)
                self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                logging.info("Connected to GR00T server at %s:%s", self.host, self.port)
                return
            except zmq.error.Again:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"GR00T server not ready at {self.host}:{self.port} "
                        f"after {timeout_s}s"
                    )
                self._init_socket()
                time.sleep(1)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        try:
            self.socket.send(_to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        response = _from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, RuntimeError):
            self._init_socket()
            return False

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        payload = {"observation": observation}
        if options is not None:
            payload["options"] = options
        response = self.call_endpoint("get_action", payload)
        if isinstance(response, (list, tuple)) and len(response) == 2:
            return response[0], response[1]
        if isinstance(response, dict) and "action" not in response:
            # Server returned the action dict directly.
            return response, {}
        raise RuntimeError(f"Unexpected get_action response type: {type(response)}")

    def reset(self, options: dict | None = None) -> dict:
        try:
            return self.call_endpoint("reset", {"options": options})
        except Exception as exc:
            logging.warning("GR00T reset failed: %s", exc)
            return {}

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self.socket = None
        try:
            self.context.term()
        except Exception:
            pass


def concat_joint_qpos(action: dict) -> np.ndarray:
    """Pack GR00T action groups into RoboTwin 14-D qpos (T, 14)."""
    keys = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    missing = [k for k in keys if k not in action]
    if missing:
        raise KeyError(f"GR00T action missing keys {missing}; got {list(action)}")
    parts = []
    for key in keys:
        arr = np.asarray(action[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"action[{key}] expected (T, D) or (B, T, D), got {arr.shape}")
        parts.append(arr)
    return np.concatenate(parts, axis=-1)
