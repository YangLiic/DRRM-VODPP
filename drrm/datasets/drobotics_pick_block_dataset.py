import copy

import numpy as np
import torch
import torch.nn.functional as F

from drrm.common.normalize_util import get_range_normalizer_from_stat
from drrm.datasets.drrm_dataset import DRRMDataset


class DRoboticsPickBlockDataset(DRRMDataset):
    """Runtime adapter for raw D-Robotics LeRobot datasets.

    The raw dataset stores low-dimensional data as ``observation.state`` and a
    broader ``action`` vector. VODPPlus training expects a top-level ``state``
    trajectory in both ``obs`` and ``action`` dicts, so this adapter builds that
    view on the fly without modifying the source dataset.
    """

    def __init__(
        self,
        action_state_indices: list[int],
        image_key_map: dict[str, str] | None = None,
        image_resize_hw: list[int] | tuple[int, int] | None = None,
        action_source_key: str = "action",
        video_backend: str | None = "pyav",
        **kwargs,
    ):
        self.action_state_indices = [int(idx) for idx in action_state_indices]
        self.image_key_map = image_key_map or {}
        self.image_resize_hw = tuple(image_resize_hw) if image_resize_hw else None
        self.action_source_key = action_source_key
        self._configured_video_backend = video_backend
        if kwargs.get("dataset_metadata") is not None:
            self._ensure_feature_names(kwargs["dataset_metadata"])
        super().__init__(**kwargs)
        self.video_backend = self._configured_video_backend

    @staticmethod
    def _ensure_feature_names(dataset_metadata):
        for key, feature in dataset_metadata.features.items():
            if "names" in feature:
                continue
            shape = feature.get("shape", [])
            if not shape:
                feature["names"] = [key]
                continue
            if len(shape) == 1:
                feature["names"] = [f"{key}_{idx}" for idx in range(shape[0])]
            else:
                feature["names"] = [f"dim_{idx}" for idx in range(len(shape))]

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
            action_state_indices=self.action_state_indices,
            image_key_map=copy.deepcopy(self.image_key_map),
            image_resize_hw=self.image_resize_hw,
            action_source_key=self.action_source_key,
            video_backend=self._configured_video_backend,
        )

    def get_normalizer(self, **kwargs):
        normalizer = super().get_normalizer(**kwargs)
        normalizer["state"] = get_range_normalizer_from_stat(
            self._selected_action_stats()
        )
        return normalizer

    def _selected_action_stats(self):
        if self._uses_all_episodes():
            stats = self.dataset_meta.stats[self.action_source_key]
            return {
                key: np.asarray(value, dtype=np.float32)[self.action_state_indices]
                for key, value in stats.items()
                if key in {"min", "max", "mean", "std"}
            }

        values = self.hf_dataset[self.action_source_key]
        if hasattr(values, "numpy"):
            values = values.numpy()
        data = np.asarray(values)
        if data.dtype == object:
            data = np.stack([np.asarray(value) for value in values])
        data = data[:, self.action_state_indices]
        return {
            "min": np.min(data, axis=0).astype(np.float32),
            "max": np.max(data, axis=0).astype(np.float32),
            "mean": np.mean(data, axis=0).astype(np.float32),
            "std": np.std(data, axis=0).astype(np.float32),
        }

    def _select_state_action(self, item):
        source = item[self.action_source_key]
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(source)
        indices = torch.as_tensor(self.action_state_indices, device=source.device)
        state_action = source.index_select(dim=-1, index=indices).to(torch.float32)
        return torch.nan_to_num(state_action, nan=0.0, posinf=0.0, neginf=0.0)

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
        state_action = self._select_state_action(item)
        data = {
            "obs": {
                "state": state_action,
            },
            "action": {
                "state": state_action.clone(),
            },
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
        return (
            f"{base}\n"
            f"Runtime adapter: action_source_key={self.action_source_key}, "
            f"state_dim={len(self.action_state_indices)}, "
            f"image_key_map={self.image_key_map}"
        )
