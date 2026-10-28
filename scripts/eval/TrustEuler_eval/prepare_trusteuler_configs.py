#!/usr/bin/env python3
"""Write TrustEuler dataset + VODPP backend JSON for one RoboTwin task/ckpt."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def _policy_rgb_keys(ckpt_cfg: dict[str, Any]) -> list[str]:
    rgb = _find_key(ckpt_cfg, "rgb_key")
    if isinstance(rgb, str):
        return [rgb]
    if isinstance(rgb, list) and rgb:
        return [str(x) for x in rgb]
    only_first = _find_key(ckpt_cfg, "only_first_view")
    obs = (ckpt_cfg.get("shape_meta") or {}).get("obs") or {}
    cams = [k for k in obs if isinstance(k, str) and k.endswith("_cam")]
    if only_first is True and "head_cam" in cams:
        return ["head_cam"]
    if cams:
        # Prefer head + wrists when present (matches common 3v RoboTwin ckpts).
        preferred = [c for c in ("head_cam", "left_cam", "right_cam") if c in cams]
        return preferred or cams
    return ["head_cam"]


def _action_dim(dataset_root: Path, ckpt_cfg: dict[str, Any]) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        feats = _read_json(info_path).get("features") or {}
        for key in ("action", "state"):
            shape = (feats.get(key) or {}).get("shape")
            if isinstance(shape, list) and shape:
                return int(shape[0])
    shape_meta = ckpt_cfg.get("shape_meta") or {}
    action = shape_meta.get("action") or {}
    shape = action.get("shape")
    if isinstance(shape, list) and shape:
        return int(shape[-1] if len(shape) > 1 else shape[0])
    raise ValueError(f"cannot infer action_dim from {dataset_root} / checkpoint")


def _horizon(ckpt_cfg: dict[str, Any]) -> int:
    horizon = _find_key(ckpt_cfg, "horizon")
    if horizon is None:
        horizon = _find_key(ckpt_cfg, "pred_horizon")
    if horizon is None:
        raise ValueError("checkpoint config missing horizon")
    return int(horizon)


def build_configs(
    *,
    task: str,
    checkpoint: Path,
    dataset_root: Path,
    drrm_root: Path,
    output_dir: Path,
    episode_count: int,
    dense_steps: int,
    use_ema: bool,
    selection_seed: int | None,
) -> tuple[Path, Path]:
    ckpt_cfg_path = checkpoint / "config.json"
    if not ckpt_cfg_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config.json: {ckpt_cfg_path}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"LeRobot dataset missing: {dataset_root}")

    ckpt_cfg = _read_json(ckpt_cfg_path)
    horizon = _horizon(ckpt_cfg)
    action_dim = _action_dim(dataset_root, ckpt_cfg)
    rgb_keys = _policy_rgb_keys(ckpt_cfg)
    cameras = {key: key for key in rgb_keys}

    if selection_seed is None:
        selection_seed = 20260731 + (zlib.crc32(task.encode("utf-8")) % 10_000)
    noise_seed = selection_seed * 100

    dataset_cfg = {
        "dataset_root": str(dataset_root.resolve()),
        "action_horizon": horizon,
        "action_key": "action",
        "state_key": "state",
        "action_slice": [0, action_dim],
        "state_slice": [0, action_dim],
        "cameras": cameras,
        "episode_count": episode_count,
        "selection_seed": selection_seed,
        "chunk_stride": horizon,
        "complete_only": True,
        "noise_seed": noise_seed,
        "noise_episode_stride": 1000,
    }
    vodpp_cfg = {
        "drrm_root": str(drrm_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "dense_steps": dense_steps,
        "use_ema": bool(use_ema),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.json"
    vodpp_path = output_dir / "vodpp.json"
    dataset_path.write_text(json.dumps(dataset_cfg, indent=2) + "\n", encoding="utf-8")
    vodpp_path.write_text(json.dumps(vodpp_cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[prepare] task={task} horizon={horizon} action_dim={action_dim} cameras={rgb_keys}")
    print(f"[prepare] dataset_cfg -> {dataset_path}")
    print(f"[prepare] vodpp_cfg   -> {vodpp_path}")
    return dataset_path, vodpp_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--drrm-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode-count", type=int, default=8)
    parser.add_argument("--dense-steps", type=int, default=100)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=None)
    args = parser.parse_args()
    build_configs(
        task=args.task,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        drrm_root=args.drrm_root,
        output_dir=args.output_dir,
        episode_count=args.episode_count,
        dense_steps=args.dense_steps,
        use_ema=args.use_ema,
        selection_seed=args.selection_seed,
    )


if __name__ == "__main__":
    main()
