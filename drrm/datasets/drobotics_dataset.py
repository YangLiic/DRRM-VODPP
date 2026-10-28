"""
D-Robotics → VODPP dataset adapter.

Converts LeRobot-format D-Robotics datasets into the batch layout expected
by VODPP training:  batch["obs"]["state"], batch["obs"]["<cam>"], batch["action"]["state"].
"""

import copy
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from drrm.common.normalize_util import get_range_normalizer_from_stat
from drrm.datasets.drrm_dataset import DRRMDataset


class DRoboticsDataset(DRRMDataset):
    """Runtime adapter for D-Robotics LeRobot datasets.

    Handles:
    - Remapping raw LeRobot image keys to model view names via ``image_key_map``.
    - Selecting/slicing state dimensions via ``state_indices``.
    - Optional on-the-fly image resize via ``image_resize_hw``.
    - Copying ``observation.state`` as the action target when no dedicated
      action column exists (``action_source_key=None``).
    - Appending extra low-dim columns (e.g. ``observation.effort``) to both the
      state observation and the action target via ``extra_source_keys``, so the
      model conditions on the current values and predicts their future
      trajectory alongside the joint action.
    - If a configured ``extra_source_key`` is missing from the LeRobot dataset,
      that column is dropped (dimension reduced) rather than zero-filled. Keep
      ``shape_meta`` state/action dims in sync with the effective state dim.
    """

    def __init__(
        self,
        state_source_key: str = "observation.state",
        action_source_key: str | None = None,
        state_indices: list[int] | None = None,
        extra_source_keys: list[str] | None = None,
        extra_source_indices: dict[str, list[int]] | None = None,
        image_key_map: dict[str, str] | None = None,
        image_resize_hw: list[int] | tuple[int, int] | None = None,
        video_backend: str | None = "pyav",
        **kwargs,
    ):
        self.state_source_key = state_source_key
        self.action_source_key = action_source_key
        self.state_indices = (
            [int(idx) for idx in state_indices] if state_indices is not None else None
        )
        self.extra_source_keys = list(extra_source_keys) if extra_source_keys else []
        self.extra_source_indices = {
            key: [int(idx) for idx in indices]
            for key, indices in (extra_source_indices or {}).items()
            if indices is not None
        }
        self.image_key_map = image_key_map or {}
        self.image_resize_hw = tuple(image_resize_hw) if image_resize_hw else None
        self._configured_video_backend = video_backend
        self._dropped_extra_source_keys: list[str] = []
        if kwargs.get("dataset_metadata") is not None:
            self._ensure_feature_names(kwargs["dataset_metadata"])
        super().__init__(**kwargs)
        self.video_backend = self._configured_video_backend
        self._resolve_extra_source_keys()

    @staticmethod
    def _ensure_feature_names(dataset_metadata):
        for key, feature in dataset_metadata.features.items():
            if "names" in feature and feature["names"] is not None:
                continue
            shape = feature.get("shape", [])
            if not shape:
                feature["names"] = [key]
            elif len(shape) == 1:
                feature["names"] = [f"{key}_{idx}" for idx in range(shape[0])]
            else:
                feature["names"] = [f"dim_{idx}" for idx in range(len(shape))]

    def _has_column(self, key: str) -> bool:
        meta = getattr(self, "dataset_meta", None)
        if meta is not None:
            features = getattr(meta, "features", None) or {}
            if key in features:
                return True
            stats = getattr(meta, "stats", None) or {}
            if key in stats:
                return True
        hf = getattr(self, "hf_dataset", None)
        if hf is not None:
            column_names = getattr(hf, "column_names", None)
            if column_names is not None and key in column_names:
                return True
        return False

    def _extra_source_dim(self, key: str) -> int | None:
        indices = self.extra_source_indices.get(key)
        if indices is not None:
            return len(indices)
        meta = getattr(self, "dataset_meta", None)
        if meta is not None:
            feature = (getattr(meta, "features", None) or {}).get(key)
            if feature is not None:
                shape = feature.get("shape") or ()
                if len(shape) >= 1:
                    return int(shape[-1])
        return None

    def _resolve_extra_source_keys(self):
        if not self.extra_source_keys:
            self._dropped_extra_source_keys = []
            return

        available = []
        missing = []
        for key in self.extra_source_keys:
            if self._has_column(key):
                available.append(key)
            else:
                missing.append(key)

        if missing:
            dropped_dims = 0
            dim_known = True
            for key in missing:
                dim = self._extra_source_dim(key)
                if dim is None:
                    dim_known = False
                else:
                    dropped_dims += dim

            dim_msg = (
                f"dropped {dropped_dims} state/action dim(s)"
                if dim_known
                else "state/action dim reduced"
            )
            warnings.warn(
                f"{self.__class__.__name__}: configured extra_source_keys {missing} "
                f"are missing from dataset {getattr(self, 'repo_id', None)!r}; "
                f"{dim_msg} (no zero-fill). Align shape_meta state/action shapes "
                f"with the effective dim (remaining extras={available}).",
                UserWarning,
                stacklevel=2,
            )

        self._dropped_extra_source_keys = missing
        self.extra_source_keys = available
        for key in missing:
            self.extra_source_indices.pop(key, None)

    def get_validation_dataset(self):
        if self.val_mask is None:
            return None

        val_episodes = np.nonzero(self.val_mask)[0]
        return self.__class__(
            repo_id=self.repo_id,
            root=self.root,
            dataset_metadata=self.dataset_meta,
            episodes=val_episodes.tolist(),
            horizon=self.horizon,
            obs_image_frame_indices=self.obs_image_frame_indices,
            rgb_key=self.rgb_key,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            npy_feature_keys=self.npy_feature_keys,
            seed=self.seed,
            val_ratio=0.0,
            max_train_episodes=None,
            cache_to_ram=False,
            cache_backend="local",
            state_source_key=self.state_source_key,
            action_source_key=self.action_source_key,
            state_indices=copy.deepcopy(self.state_indices),
            extra_source_keys=copy.deepcopy(self.extra_source_keys),
            extra_source_indices=copy.deepcopy(self.extra_source_indices),
            image_key_map=copy.deepcopy(self.image_key_map),
            image_resize_hw=self.image_resize_hw,
            video_backend=self._configured_video_backend,
        )

    def get_normalizer(self, **kwargs):
        normalizer = super().get_normalizer(**kwargs)
        stat = self._selected_training_action_stats()
        for key in self.extra_source_keys:
            extra_stat = self._column_stats(key, self.extra_source_indices.get(key))
            stat = {
                name: np.concatenate([stat[name], extra_stat[name]], axis=-1)
                for name in stat
            }
        normalizer["state"] = get_range_normalizer_from_stat(stat)
        return normalizer

    def _selected_training_action_stats(self):
        source_key = self.action_source_key or self.state_source_key
        return self._column_stats(source_key, self.state_indices)

    def _column_stats(self, source_key, indices):
        if self._uses_all_episodes():
            stats = self.dataset_meta.stats[source_key]
            result = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.items()
                if key in {"min", "max", "mean", "std"}
            }
        else:
            values = self.hf_dataset[source_key]
            if hasattr(values, "numpy"):
                values = values.numpy()
            data = np.asarray(values)
            if data.dtype == object:
                data = np.stack([np.asarray(value) for value in values])
            result = {
                "min": np.min(data, axis=0).astype(np.float32),
                "max": np.max(data, axis=0).astype(np.float32),
                "mean": np.mean(data, axis=0).astype(np.float32),
                "std": np.std(data, axis=0).astype(np.float32),
            }
        if indices is not None:
            result = {key: value[..., indices] for key, value in result.items()}
        return result

    def _select_vector(self, source):
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(source)
        source = source.to(torch.float32)
        if self.state_indices is not None:
            indices = torch.as_tensor(self.state_indices, device=source.device)
            source = source.index_select(dim=-1, index=indices)
        return torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_state(self, item):
        return self._select_vector(item[self.state_source_key])

    def _select_action(self, item, state):
        if self.action_source_key is None:
            return state.clone()
        return self._select_vector(item[self.action_source_key])

    def _select_extras(self, item):
        extras = []
        for key in self.extra_source_keys:
            source = item[key]
            if not isinstance(source, torch.Tensor):
                source = torch.as_tensor(source)
            source = source.to(torch.float32)
            indices = self.extra_source_indices.get(key)
            if indices is not None:
                index = torch.as_tensor(indices, device=source.device)
                source = source.index_select(dim=-1, index=index)
            extras.append(
                torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            )
        return extras

    def _resize_image(self, image):
        if self.image_resize_hw is None:
            return image
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        image = image.to(torch.float32)
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        if image.numel() > 0 and image.max() > 2:
            image = image / 255.0

        original_ndim = image.ndim
        if original_ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        resized = F.interpolate(
            image,
            size=self.image_resize_hw,
            mode="bilinear",
            align_corners=False,
        )
        resized = torch.nan_to_num(resized, nan=0.0, posinf=1.0, neginf=0.0)
        return resized if original_ndim == 3 else resized

    def _organize_item_data(self, item):
        state = self._select_state(item)
        action = self._select_action(item, state)
        extras = self._select_extras(item)
        if extras:
            # Efforts (etc.) join both vectors: obs frames condition the model
            # on the current values, action frames supervise their prediction.
            state = torch.cat([state, *extras], dim=-1)
            action = torch.cat([action, *extras], dim=-1)
        data = {
            "obs": {"state": state},
            "action": {"state": action},
        }

        for raw_key, train_key in self.image_key_map.items():
            if raw_key in item:
                data["obs"][train_key] = self._resize_image(item[raw_key])

        if self.detail_item:
            for key in ["episode_index", "frame_index", "task_index"]:
                if key in item:
                    data["obs"][key] = item[key]

        return data

    def __repr__(self):
        base = super().__repr__()
        dim = "all" if self.state_indices is None else len(self.state_indices)
        dropped = getattr(self, "_dropped_extra_source_keys", [])
        return (
            f"{base}\n"
            f"Runtime adapter: state_source_key={self.state_source_key}, "
            f"action_source_key={self.action_source_key or 'state copy'}, "
            f"state_dim={dim}, extra_source_keys={self.extra_source_keys}, "
            f"dropped_extra_source_keys={dropped}, "
            f"image_key_map={self.image_key_map}"
        )


# Backward-compatible aliases for existing checkpoints / yaml configs.
OneBoxPickDataset = DRoboticsDataset
DualBoxPickDataset = DRoboticsDataset
