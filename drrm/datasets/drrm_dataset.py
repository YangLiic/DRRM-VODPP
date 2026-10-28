import copy
from bisect import bisect_right

import torch

from drrm.datasets.episode_cache import LocalEpisodeCacheBackend
from drrm.datasets.shm_episode_cache import ShmEpisodeCacheClient
from drrm.datasets.video_frame_decoder import (
    decode_all_video_frames_pyav,
    decode_video_frames_pyav_seek,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from pathlib import Path
import numpy as np
import shutil
from lerobot.datasets.compute_stats import sample_images, get_feature_stats
from lerobot.datasets.utils import (
    validate_episode_buffer,
    get_episode_data_index,
    check_timestamps_sync,
)
from drrm.common.normalizer import LinearNormalizer
from drrm.common.normalize_util import (
    get_image_range_normalizer, 
    get_range_normalizer_from_stat, 
    get_identity_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat
)

def downsample_mask(mask, max_n, seed=0):
    """Downsample training data to max_n samples"""
    if max_n is None or np.sum(mask) <= max_n:
        return mask
    
    n_train = int(max_n)
    curr_train_idxs = np.nonzero(mask)[0]
    rng = np.random.default_rng(seed=seed)
    train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
    train_idxs = curr_train_idxs[train_idxs_idx]
    
    new_mask = np.zeros_like(mask)
    new_mask[train_idxs] = True
    return new_mask


def get_val_mask(n_episodes, val_ratio, seed=0):
    """Create validation mask for episodes"""
    if val_ratio <= 0:
        return np.zeros(n_episodes, dtype=bool)

    # Ensure at least 1 episode for validation and 1 for training
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    
    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_idxs] = True
    return val_mask


def compute_episode_stats(episode_data, features):
    """Compute statistics for episode data"""
    ep_stats = {}
    
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            if len(features[key]['shape']) >= 2:
                ep_ft_array = data  # data is already a np.ndarray
                axes_to_reduce = tuple(range(len(data.shape) - 1)) # compute stats over the first axis
                keepdims = True 
            else:
                ep_ft_array = data  # data is already a np.ndarray
                axes_to_reduce = 0  # compute stats over the first axis
                keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats

class DRRMDataset(LeRobotDataset):
    """Enhanced LeRobot dataset with DRRM-specific features"""
    
    DEFAULT_KEYS = {'timestamp', 'frame_indx', 'episode_index', 'index', 'task_index', 'frame_index'}
    FILTER_KEYS = {'timestamp', 'frame_indx', 'index'}

    def __init__(self, 
            repo_id: str,
            root:  str | Path | None = None,
            dataset_metadata: LeRobotDatasetMetadata | None = None,
            episodes: list[int] | None = None,
            task_list: list[int] | None = None,
            task_index_list: list[int] | None = None,
            black_index: list[int] | None = None,
            horizon: int = 8,
            obs_image_frame_indices: list[int] | None = None,
            rgb_key: list[str] | str | None = None,
            pad_before: int = 0,
            pad_after: int = 0,
            npy_feature_keys: list[str] | None = None,
            seed: int = 0,
            val_ratio: float = 0.0,
            max_train_episodes=None,
            cache_to_ram: bool = False,
            cache_mode: str = "lazy",
            cache_backend: str = "local",
            cache_socket: str | None = None,
            cache_max_episodes: int | None = None,
            cache_report_interval: int = 1000,
            cache_decoded_video: bool = False,
            **kwargs
        ):
        
        # Store all parameters for easy access
        self.dataset_meta = dataset_metadata
        self.repo_id = repo_id
        self.root = root
        self.horizon = horizon
        self.obs_image_frame_indices = obs_image_frame_indices
        self.rgb_key = [rgb_key] if isinstance(rgb_key, str) else rgb_key
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.npy_feature_keys = npy_feature_keys or []
        self.val_mask = None
        self.task_list = task_list
        self.task_index_list = task_index_list
        self.black_index = black_index
        self.detail_item = kwargs.pop('detail_item', False)
        self.cache_to_ram = cache_to_ram
        self.cache_mode = cache_mode
        self.cache_backend = cache_backend
        self.cache_socket = cache_socket
        self.cache_max_episodes = cache_max_episodes
        self.cache_report_interval = cache_report_interval
        # Opt-in: also decode every video frame of a cached episode once (at
        # cache build time) and store it alongside the other RAM/SHM-cached
        # columns, so later epochs re-read the decoded frame instead of
        # re-decoding it from the compressed video every time. This trades
        # RAM for CPU: a fully-decoded episode can be 1-2 orders of magnitude
        # larger than its compressed .mp4, so prefer `cache_max_episodes` to
        # bound memory use on large video datasets.
        self.cache_decoded_video = cache_decoded_video
        self._episode_cache = None
        self._episode_cache_backend = None
        self._episode_cache_hits = 0
        self._episode_cache_misses = 0
        self._episode_cache_evictions = 0
        if self.cache_mode not in {"lazy", "eager"}:
            raise ValueError(f"Unsupported cache_mode: {self.cache_mode}")
        if self.cache_backend not in {"local", "shm"}:
            raise ValueError(f"Unsupported cache_backend: {self.cache_backend}")
        
        # Setup delta timestamps
        self.delta_timestamps = self._create_delta_timestamps()
        
        # Handle episode selection and train/val split
        if episodes is None:
            episodes = self._setup_train_val_split()
        episodes = [int(ep_idx) for ep_idx in episodes]
        self.selected_episodes = episodes
            
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)}
        
        # Initialize parent class
        super().__init__(
            repo_id=repo_id, 
            root=root, 
            episodes=episodes,
            delta_timestamps=self.delta_timestamps
        )
        self.base_hf_dataset = self._create_base_hf_dataset()
        self._build_episode_cache_index()
        if self.cache_to_ram:
            self._setup_episode_cache()

    def _create_delta_timestamps(self):
        """Create delta timestamps for all non-default keys"""
        timestamps = {}
        for key in self.dataset_meta.names.keys():
            if key not in self.DEFAULT_KEYS:
                feature = self.dataset_meta.features[key]
                if feature["dtype"] in {"image", "video"} and not self._is_selected_visual_key(key):
                    continue
                if feature["dtype"] in {"image", "video"}:
                    frame_indices = [
                        t - self.pad_before
                        for t in self.obs_image_frame_indices
                    ]
                else:
                    frame_indices = range(-self.pad_before, -self.pad_before + self.horizon)
                timestamps[key] = [
                    t / self.dataset_meta.fps
                    for t in frame_indices
                ]
        return timestamps

    def _is_selected_visual_key(self, key):
        rgb_key = getattr(self, "rgb_key", None)
        return rgb_key is None or key in rgb_key

    def _is_cacheable_episode_key(self, key):
        image_keys = set(getattr(self.meta, "image_keys", []))
        video_keys = set(getattr(self.meta, "video_keys", []))
        if key in video_keys:
            return False
        if key in image_keys:
            return self._is_selected_visual_key(key)
        return True

    def _filter_episode_cache_rows(self, rows):
        return {
            key: values
            for key, values in rows.items()
            if self._is_cacheable_episode_key(key)
        }

    def _create_base_hf_dataset(self):
        """Use scalar/non-camera columns for base row reads; camera chunks are queried separately."""
        camera_keys_to_remove = [
            key for key in getattr(self.meta, "camera_keys", [])
            if key in self.hf_dataset.column_names
        ]
        if not camera_keys_to_remove:
            return self.hf_dataset
        return self.hf_dataset.remove_columns(camera_keys_to_remove)

    def _setup_train_val_split(self):
        """Setup train/validation split and return train episodes"""
        tn_episodes = self.dataset_meta.total_episodes
        available_mask = np.zeros(tn_episodes, dtype=bool)
        self.train_mask = np.zeros(tn_episodes, dtype=bool)
        # task list -> task index list
        if self.task_index_list is not None:
            if self.task_list is None:
                self.task_list = []
            self.task_list += [self.dataset_meta.tasks[ind] for ind in self.task_index_list]
        if self.task_list is None:
            self.task_list = list(self.dataset_meta.tasks.values())
        # task index list -> available_mask
        if 'task' in ','.join(self.dataset_meta.episodes[0].keys()):
            assert not 'task' in self.dataset_meta.episodes[0], \
                "FORMAT ERRO: detective `task` key in dataset_meta, please change it to `tasks` in list type"
            available_mask[[
                bool(set(ep['tasks']) & set(self.task_list))
                # ep['tasks'][0] in set(self.task_list)
                for ep in self.dataset_meta.episodes.values()
            ]] = True
        else:
            available_mask[:] = True
        if self.black_index is not None:
            available_mask[self.black_index] = False

        
        n_episodes = available_mask.sum()
        val_mask = get_val_mask(n_episodes, self.val_ratio, self.seed)
        if val_mask.sum() > 0:
            self.val_mask = np.zeros(tn_episodes, dtype=bool)
            self.val_mask[available_mask] = val_mask
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, self.max_train_episodes, self.seed)
        self.train_mask[available_mask] = train_mask
        return np.nonzero(self.train_mask)[0]

    def get_validation_dataset(self):
        """Create validation dataset if validation split exists"""
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
            val_ratio=0.0,  # No further splitting for validation
            max_train_episodes=None,
            cache_to_ram=False,
            cache_backend="local",
        )

    def get_normalizer(self, **kwargs):
        """Create normalizer for all dataset features"""
        normalizer = LinearNormalizer()
        
        # Add chunk-wise delta feature normalizers
        if 'endpose' in self.dataset_meta.names.keys():
            normalizer['delta_endpose'] = self._compute_delta_stats('delta_endpose')
        if 'state' in self.dataset_meta.names.keys():
            normalizer['delta_state'] = self._compute_delta_stats('delta_state')
        
        # Add VGGT feature normalizers (identity)
        for key in self.npy_feature_keys:
            normalizer[key] = self._create_identity_normalizer()
        
        # Add dataset feature normalizers
        for key in self.dataset_meta.names.keys():
            if key in self.DEFAULT_KEYS:
                continue
                
            feature = self.dataset_meta.features[key]
            normalizer[key] = self._create_feature_normalizer(key, feature)
                
        return normalizer

    def _compute_delta_stats(self, key, num_samples=1000):
        """Compute mean, std, min, max for dynamically generated delta features."""
        # Randomly sample at most num_samples to compute statistics to avoid slow down
        n = len(self)
        sample_size = min(n, num_samples)
        
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(n, size=sample_size, replace=False)
        
        base_key = key.replace('delta_', '')
        if base_key not in self.hf_dataset.column_names:
            return self._create_identity_normalizer()
        
        # 1. Only read episode ids; avoid materializing image/video/heavy columns.
        episode_dataset = self.hf_dataset.select_columns(["episode_index"])
        batch_items = episode_dataset[indices.tolist()]
        episode_indices = batch_items["episode_index"]
        
        # 2. 计算所有的 query 帧索引
        all_chunk_idx = []
        chunk_size = 1
        for i, idx in enumerate(indices):
            ep_idx = episode_indices[i].item() if hasattr(episode_indices[i], 'item') else episode_indices[i]
            if getattr(self, "delta_indices", None) is not None:
                # 获取该序列的动作切片位置
                q_inds, _ = getattr(self, "_get_query_indices")(int(idx), self.ep_idx_to_arr_idx[ep_idx])
                chunk_indices = q_inds["action"]
                all_chunk_idx.extend(chunk_indices)
                chunk_size = len(chunk_indices)
            else:
                all_chunk_idx.append(int(idx))
                
        if not all_chunk_idx:
            return self._create_identity_normalizer()
            
        # 3. Deduplicate repeated frame ids before the expensive HF dataset read.
        all_chunk_idx = np.asarray(all_chunk_idx, dtype=np.int64)
        unique_chunk_idx, inverse = np.unique(all_chunk_idx, return_inverse=True)
        base_dataset = self.hf_dataset.select_columns([base_key])
        unique_base_data = base_dataset[unique_chunk_idx.tolist()][base_key]
        unique_base_data = np.asarray(unique_base_data)
        if unique_base_data.dtype == object:
            unique_base_data = np.stack([np.asarray(value) for value in unique_base_data])
        flat_base_data = unique_base_data[inverse]
        
        # 4. Reshape并计算相对差值
        if getattr(self, "delta_indices", None) is not None:
            # 形状: [N, T, Dim]
            base_data = flat_base_data.reshape(len(indices), chunk_size, -1)
        else:
            base_data = flat_base_data.reshape(len(indices), 1, -1)
        
        all_data = base_data.copy()
        if key.startswith('delta_'):
            # 广播减去第一帧以获得相对差
            all_data[:, 1:, :] = all_data[:, 1:, :] - all_data[:, 0:1, :]
            
        # 提取统计特征，注意多维情况下的axis
        stat = {
            'min': np.min(all_data, axis=(0)).astype(np.float32),
            'max': np.max(all_data, axis=(0)).astype(np.float32),
            'mean': np.mean(all_data, axis=(0)).astype(np.float32),
            'std': np.std(all_data, axis=(0)).astype(np.float32),
        }
        # Safeguard for zero std
        stat['std'] = np.clip(stat['std'], 1e-4, None)
        
        if 'endpose' in base_key:
            return robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            
        return get_range_normalizer_from_stat(stat)

    def _create_identity_normalizer(self):
        """Create identity normalizer with wide bounds"""
        stat = {
            'min': np.float32([-1e8]),
            'max': np.float32([1e8]),
            'mean': np.float32([0]),
            'std': np.float32([1e8]),
        }
        return get_identity_normalizer_from_stat(stat)

    def _create_feature_normalizer(self, key, feature):
        """Create appropriate normalizer based on feature type"""
        if feature['dtype'] in ['image', 'video']:
            return get_image_range_normalizer()
        elif feature['dtype'] == 'string':
            return self._create_identity_normalizer()
        else:
            stat = self._get_feature_stats_for_normalizer(key)
            if 'endpose' in key:
                return robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            return get_range_normalizer_from_stat(stat)

    def _uses_all_episodes(self):
        """Return whether the current dataset covers all episodes in metadata."""
        selected = getattr(self, 'selected_episodes', None)
        if selected is None:
            return True

        total_episodes = self.dataset_meta.total_episodes
        if len(selected) != total_episodes:
            return False

        return set(selected) == set(range(total_episodes))

    def _get_feature_stats_for_normalizer(self, key):
        """Use filtered dataset stats when this dataset is an episode subset."""
        if self._uses_all_episodes():
            stats = self.dataset_meta.stats[key]
            return {
                'min': stats['min'].astype(np.float32),
                'max': stats['max'].astype(np.float32),
                'mean': stats['mean'].astype(np.float32),
                'std': stats['std'].astype(np.float32),
            }

        values = self.hf_dataset[key]
        if hasattr(values, 'numpy'):
            values = values.numpy()
        data = np.asarray(values)
        if data.dtype == object:
            data = np.stack([np.asarray(value) for value in values])

        return {
            'min': np.min(data, axis=0).astype(np.float32),
            'max': np.max(data, axis=0).astype(np.float32),
            'mean': np.mean(data, axis=0).astype(np.float32),
            'std': np.std(data, axis=0).astype(np.float32),
        }
        
    def _get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]] | None = None) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in getattr(self.meta, "video_keys", []):
            if not self._is_selected_visual_key(key):
                continue
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = self._lookup_timestamps(query_indices[key]).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _lookup_timestamps(self, indices):
        """Resolve row indices to their `timestamp` column values.

        `Dataset.select()` rebuilds a whole new HF `Dataset` object (schema/
        fingerprint deepcopy and all) just to read a handful of scalars, which
        dominates `__getitem__` cost once per video key when video_keys are
        present (independent of video decoding itself). Route through the
        episode row cache first, since `timestamp` is already cached like any
        other non-visual column; only fall back to `.select()` on a cache miss.
        """
        cached = self._query_hf_dataset_cached({"timestamp": list(indices)})
        timestamps = cached.get("timestamp") if cached else None
        if timestamps is None:
            timestamps = torch.stack(self.hf_dataset.select(list(indices))["timestamp"])
        return timestamps

    def _query_videos(self, query_timestamps, ep_idx, query_indices=None):
        """Query the video frames needed by the current training item.

        Prefers frames already decoded into the episode RAM/SHM cache (see
        `_decode_episode_videos`/`cache_decoded_video`) over decoding from the
        compressed video, falling back per-view to on-the-fly decode for any
        view that isn't cached (cache disabled, not yet built, or evicted).
        """
        remaining_timestamps = dict(query_timestamps)
        cached_frames = {}

        if self.cache_decoded_video and self._episode_cache is not None and query_indices:
            for video_key in list(remaining_timestamps.keys()):
                indices = query_indices.get(video_key)
                if not indices:
                    continue
                frames = self._get_cached_video_frames(video_key, indices, ep_idx)
                if frames is not None:
                    cached_frames[video_key] = frames
                    del remaining_timestamps[video_key]

        if not remaining_timestamps:
            return cached_frames

        if getattr(self, "video_backend", None) not in {"pyav", "pyav_seek"}:
            decoded = super()._query_videos(remaining_timestamps, ep_idx)
        else:
            decoded = {}
            for video_key, timestamps in remaining_timestamps.items():
                video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
                frames = decode_video_frames_pyav_seek(
                    video_path=video_path,
                    timestamps=timestamps,
                    tolerance_s=self.tolerance_s,
                )
                # Match LeRobot's output convention for a single requested frame.
                decoded[video_key] = frames.squeeze(0)

        cached_frames.update(decoded)
        return cached_frames
    
    def _query_npy_frame(self, query_indices, ep_idx):
        """Load npy features from numpy files"""
        if not self.npy_feature_keys or query_indices is None:
            return {}
            
        relative_ep_idx = self.ep_idx_to_arr_idx[ep_idx]
        indices = [
            idx - self.episode_data_index["from"][relative_ep_idx].item() 
            for idx in query_indices['action']
        ]
        
        result = {}
        lerobot_root = self.dataset_meta.root
        
        for key in self.npy_feature_keys:
            file_path = lerobot_root / f"npy/{key}/{ep_idx}.npy"
            if not file_path.exists():
                file_path = lerobot_root / f"npy/{key}/episode_{ep_idx:06d}.npy"
            # 使用mmeap_mode + take
            with open(file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                offset = f.tell()
            arr_memmap = np.memmap(file_path, dtype=dtype, mode='r', offset=offset).reshape(shape)
            result[key] = np.take(arr_memmap, indices, axis=0)
            
        return result

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _as_int(self, value):
        return int(value.item()) if hasattr(value, "item") else int(value)

    def _build_episode_cache_index(self):
        ranges = []
        for ep_idx in self.selected_episodes:
            arr_idx = self.ep_idx_to_arr_idx[int(ep_idx)]
            start = self._as_int(self.episode_data_index["from"][arr_idx])
            end = self._as_int(self.episode_data_index["to"][arr_idx])
            ranges.append((start, end, int(ep_idx)))

        ranges.sort(key=lambda item: item[0])
        self._episode_ranges = {ep_idx: (start, end) for start, end, ep_idx in ranges}
        self._episode_starts = [start for start, _, _ in ranges]
        self._episode_ends = [end for _, end, _ in ranges]
        self._episode_ids = [ep_idx for _, _, ep_idx in ranges]

    def _setup_episode_cache(self):
        """Initialize the configured episode RAM cache backend."""
        if not hasattr(self, "_episode_ranges"):
            self._build_episode_cache_index()
        if self.cache_backend == "shm":
            if not self.cache_socket:
                raise ValueError("cache_socket is required when cache_backend=shm")
            self._episode_cache_backend = ShmEpisodeCacheClient(self.cache_socket)
            self._episode_cache = self._episode_cache_backend
            return

        self._episode_cache_backend = LocalEpisodeCacheBackend(
            dataset=self,
            mode=self.cache_mode,
            max_episodes=self.cache_max_episodes,
            report_interval=self.cache_report_interval,
            in_worker_fn=self._in_dataloader_worker,
        )
        self._episode_cache = self._episode_cache_backend.cache

    def _get_episode_idx_for_item(self, idx):
        item_idx = int(idx)
        range_idx = bisect_right(self._episode_starts, item_idx) - 1
        if range_idx < 0 or item_idx >= self._episode_ends[range_idx]:
            raise IndexError(f"Dataset index {idx} is outside selected episode ranges")
        return self._episode_ids[range_idx]

    def _load_episode_to_ram(self, ep_idx):
        start, end = self._episode_ranges[int(ep_idx)]
        indices = list(range(start, end))
        rows = self._filter_episode_cache_rows(self.hf_dataset[indices])
        if self.cache_decoded_video:
            rows.update(self._decode_episode_videos(ep_idx, num_rows=end - start))
        return {"start": start, "end": end, "rows": rows}

    def _cacheable_video_keys(self):
        return [
            key
            for key in getattr(self.meta, "video_keys", [])
            if self._is_selected_visual_key(key)
        ]

    def _decode_episode_videos(self, ep_idx, num_rows):
        """Fully decode the selected camera views of one episode, once.

        This is only called while building the episode RAM/SHM cache
        (``cache_decoded_video=True``). Doing the seek-free, sequential
        decode here means every later ``__getitem__`` for this episode -
        across every training epoch - can be served from the cache instead
        of repeating a compressed-video seek+decode.
        """
        video_keys = self._cacheable_video_keys()
        if not video_keys:
            return {}
        result = {}
        for video_key in video_keys:
            video_path = self.root / self.meta.get_video_file_path(int(ep_idx), video_key)
            frames = decode_all_video_frames_pyav(video_path, expected_num_frames=num_rows)
            result[video_key] = list(frames.unbind(0))
        return result

    def _get_cached_video_frames(self, video_key, indices, ep_idx):
        """Look up already-decoded frames for `indices` (row-indexed). None on cache miss."""
        if self.cache_backend == "shm":
            try:
                rows = self._episode_cache_backend.get_rows({video_key: indices})
            except KeyError:
                return None
            return torch.stack(rows[video_key]).squeeze(0)

        episode = self._get_cached_episode(ep_idx)
        if episode is None:
            return None
        rows = episode["rows"]
        if video_key not in rows:
            return None
        start = int(episode["start"])
        try:
            frames = [rows[video_key][int(idx) - start] for idx in indices]
        except IndexError:
            return None
        return torch.stack(frames).squeeze(0)

    def _in_dataloader_worker(self):
        return torch.utils.data.get_worker_info() is not None

    def _get_cached_episode(self, ep_idx):
        if self._episode_cache_backend is None:
            return None
        episode = self._episode_cache_backend.get(ep_idx)
        if self.cache_backend == "local":
            stats = self._episode_cache_backend.stats()
            self._episode_cache_hits = stats["hits"]
            self._episode_cache_misses = stats["misses"]
            self._episode_cache_evictions = stats["evictions"]
        return episode

    def _row_from_episode(self, episode, idx, include_images=True):
        local_idx = int(idx) - int(episode["start"])
        rows = episode["rows"]
        image_keys = set(getattr(self.meta, "image_keys", []))
        return {
            key: values[local_idx]
            for key, values in rows.items()
            if include_images or key not in image_keys
        }

    def _query_episode_cached_row(self, idx, include_images=True):
        if self._episode_cache is None:
            return self.hf_dataset[int(idx)]
        ep_idx = self._get_episode_idx_for_item(idx)
        if self.cache_backend == "shm":
            try:
                return self._episode_cache_backend.get_row(
                    idx,
                    include_images=include_images,
                    image_keys=getattr(self.meta, "image_keys", []),
                )
            except KeyError:
                return self.hf_dataset[int(idx)]
        episode = self._get_cached_episode(ep_idx)
        if episode is None:
            return self.hf_dataset[int(idx)]
        return self._row_from_episode(episode, idx, include_images=include_images)

    def _query_base_item(self, idx):
        if self._episode_cache is None:
            return self.base_hf_dataset[idx]
        ep_idx = self._get_episode_idx_for_item(idx)
        if self.cache_backend == "shm":
            try:
                return self._episode_cache_backend.get_row(
                    idx,
                    include_images=False,
                    image_keys=getattr(self.meta, "image_keys", []),
                )
            except KeyError:
                return self.base_hf_dataset[int(idx)]
        episode = self._get_cached_episode(ep_idx)
        if episode is None:
            return self.base_hf_dataset[int(idx)]
        return self._row_from_episode(episode, idx, include_images=False)

    def _query_hf_dataset_cached(self, query_indices):
        if self._episode_cache is None:
            return super()._query_hf_dataset(query_indices)

        all_indices = [
            int(idx)
            for key, indices in query_indices.items()
            if key not in self.meta.video_keys
            for idx in indices
        ]
        if not all_indices:
            return {}

        if self.cache_backend == "shm":
            filtered_indices = {
                key: indices
                for key, indices in query_indices.items()
                if key not in self.meta.video_keys
            }
            try:
                rows = self._episode_cache_backend.get_rows(filtered_indices)
            except KeyError:
                return super()._query_hf_dataset(query_indices)
            return {key: torch.stack(values) for key, values in rows.items()}

        episode_ids = {self._get_episode_idx_for_item(idx) for idx in all_indices}
        if len(episode_ids) != 1:
            return super()._query_hf_dataset(query_indices)

        ep_idx = next(iter(episode_ids))
        episode = self._get_cached_episode(ep_idx)
        if episode is None:
            return super()._query_hf_dataset(query_indices)
        start = int(episode["start"])
        rows = episode["rows"]
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            if key not in rows:
                return super()._query_hf_dataset(query_indices)
            result[key] = torch.stack([rows[key][int(idx) - start] for idx in indices])
        return result

    def __getitem__(self, idx):
        """Get dataset item with all features"""
        # Get base item from HuggingFace dataset
        item = self._query_base_item(idx) # idx是新数据集上的index
        ep_idx = item["episode_index"].item() # 根据item["episode_index"]找到旧数据集上的episode_index
        
        # Query temporal data if needed
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, self.ep_idx_to_arr_idx[ep_idx])
            query_result = self._query_hf_dataset_cached(query_indices)
            item.update(padding)
            item.update(query_result)

        # Add npy features from numpy files
        npy_frame = self._query_npy_frame(query_indices, ep_idx)
        item.update(npy_frame)
        
        # Add video frames if needed
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx, query_indices=query_indices)
            item.update(video_frames)

        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        # Add task string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        # Calculate chunk-wise deltas for continuous control features
        if 'endpose' in item:
            # chunk-wise delta: frame 1..T are differences relative to frame 0
            item['delta_endpose'] = item['endpose'] - item['endpose'][0:1]
            item['delta_endpose'][0:1] = item['endpose'][0:1]
        if 'state' in item:
            item['delta_state'] = item['state'] - item['state'][0:1]
            item['delta_state'][0:1] = item['state'][0:1]
        
        # Organize into expected output format
        return self._organize_item_data(item)

    def _organize_item_data(self, item):
        """Organize raw item data into expected format"""
        # data = {
        #     'obs': {},
        #     'action': item['action'],
        # }
        data = {
            'obs': {},
            'action': {},
        }
        
        for k in ['action', 'state', 'delta_state', 'endpose', 'delta_endpose']:
            if k in item:
                data['action'][k] = item.get(k)

        # Add features to observations
        filter_keys = {*self.FILTER_KEYS}
        if not self.detail_item:
            filter_keys.update(self.DEFAULT_KEYS)
            
        valid_keys = {*self.npy_feature_keys, *self.dataset_meta.names.keys(), 'delta_endpose', 'delta_state'}
        for key in item:
            if key in valid_keys and key not in {*filter_keys, 'action'}:
                data['obs'][key] = item[key]
                
        return data
