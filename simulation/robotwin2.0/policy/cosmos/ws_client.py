"""Minimal OpenPI websocket client (msgpack-numpy). No openpi server deps required."""

from __future__ import annotations

import logging
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = msgpack.Packer(default=_pack_array)


class WebsocketClientPolicy:
    """Talk to openpi `scripts/serve_policy.py` over websocket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._ws, self.metadata = self._connect()

    def _connect(self):
        logging.info("Waiting for openpi server at %s ...", self._uri)
        while True:
            try:
                # First pi0.5 infer can take minutes (compile/warmup); disable keepalive ping.
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    open_timeout=600,
                    close_timeout=600,
                )
                metadata = msgpack.unpackb(conn.recv(), object_hook=_unpack_array)
                logging.info("Connected to openpi server at %s", self._uri)
                return conn, metadata
            except ConnectionRefusedError:
                time.sleep(2)

    def infer(self, obs: dict) -> dict:
        self._ws.send(_packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"openpi server error:\n{response}")
        return msgpack.unpackb(response, object_hook=_unpack_array)
