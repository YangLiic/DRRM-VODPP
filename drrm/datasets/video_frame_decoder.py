"""Targeted video-frame decoding helpers used by training datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def decode_video_frames_pyav_seek(
    video_path: str | Path,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode only the compressed span needed for ``timestamps``.

    The stock LeRobot pyav path currently goes through torchvision's
    ``VideoReader``.  Keeping the seek/stop logic here makes the training
    behavior explicit: seek to the closest preceding keyframe, stop as soon as
    the last requested timestamp is reached, and convert only the selected
    frames to RGB tensors.
    """
    if not timestamps:
        raise ValueError("At least one video timestamp is required")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - pyav is a runtime dependency
        raise ImportError("The 'pyav' video backend requires the 'av' package") from exc

    query_timestamps = [float(timestamp) for timestamp in timestamps]
    first_timestamp = min(query_timestamps)
    last_timestamp = max(query_timestamps)
    closest_frames = [None] * len(query_timestamps)
    closest_distances = [float("inf")] * len(query_timestamps)
    closest_timestamps = [None] * len(query_timestamps)

    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise RuntimeError(f"Video stream has no time base: {video_path}")

        # thread_count=0 (the PyAV/FFmpeg default) asks the decoder to pick a
        # thread count automatically, which on high-core-count hosts spins up
        # far more threads than a single tiny random-access frame needs and
        # the spin-up/sync overhead dominates. Pinning a small explicit count
        # avoids that. Measured ~15x per-frame decode speedup on AV1-encoded
        # episodes (e.g. ~38ms -> ~2.5ms per random-access frame).
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 2

        # PyAV expects stream-relative integer PTS units when a stream is given.
        # Seeking backward to a keyframe is required for inter-frame codecs.
        seek_pts = max(0, int(first_timestamp / float(stream.time_base)))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_timestamp = float(frame.pts * stream.time_base)

            for query_idx, query_timestamp in enumerate(query_timestamps):
                distance = abs(frame_timestamp - query_timestamp)
                if distance < closest_distances[query_idx]:
                    closest_distances[query_idx] = distance
                    closest_timestamps[query_idx] = frame_timestamp
                    closest_frames[query_idx] = frame

            # The first frame at or after the largest query is sufficient to
            # decide the nearest frame for every requested timestamp.
            if frame_timestamp >= last_timestamp:
                break

        if any(frame is None for frame in closest_frames):
            raise RuntimeError(
                f"Could not decode frames for timestamps {query_timestamps} from {video_path}"
            )

        invalid = [
            distance
            for distance in closest_distances
            if distance >= float(tolerance_s)
        ]
        if invalid:
            raise AssertionError(
                "One or more decoded video frames violate the timestamp "
                f"tolerance ({invalid} >= tolerance_s={tolerance_s}). "
                f"queried timestamps: {query_timestamps}; "
                f"loaded timestamps: {closest_timestamps}; video: {video_path}"
            )

        # Intermediate codec frames are never converted to RGB arrays.  When
        # queries select the same frame, convert it only once.
        converted_frames = {}
        output_frames = []
        for frame in closest_frames:
            frame_id = id(frame)
            tensor = converted_frames.get(frame_id)
            if tensor is None:
                array = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255)
                converted_frames[frame_id] = tensor
            output_frames.append(tensor)

    return torch.stack(output_frames)


def decode_all_video_frames_pyav(
    video_path: str | Path,
    expected_num_frames: int | None = None,
) -> torch.Tensor:
    """Sequentially decode every frame of a video container, in presentation order.

    Intended for building the episode RAM/SHM cache: a single straight-through
    decode pass (no seeking) over a whole episode is far cheaper than repeating
    a seek-and-decode for every individual frame access, and lets the decoded
    result be reused across every subsequent epoch instead of being recomputed
    from the compressed video every time.

    Frames are returned as ``uint8`` (channel-first, 0-255) rather than the
    ``float32``/[0, 1] convention used by :func:`decode_video_frames_pyav_seek`
    to keep the cached footprint 4x smaller -- decoding a whole dataset's
    videos can otherwise dwarf available RAM. Downstream consumers (e.g.
    ``DRoboticsDataset._resize_image``) already normalize either convention.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - pyav is a runtime dependency
        raise ImportError("The 'pyav' video backend requires the 'av' package") from exc

    arrays = []
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]

        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 2

        for frame in container.decode(stream):
            arrays.append(frame.to_ndarray(format="rgb24"))

    if expected_num_frames is not None and len(arrays) != expected_num_frames:
        raise RuntimeError(
            f"Decoded {len(arrays)} frames from {video_path}, expected "
            f"{expected_num_frames} (one per episode row)."
        )

    # Converting each frame to a torch tensor inside the loop (permute +
    # contiguous per frame) was measured to be ~45x slower than doing it once
    # over the whole stacked batch: PyTorch's default CPU intra-op thread
    # pool (sized to all visible cores) re-synchronizes its worker threads on
    # every tiny per-op call, and that synchronization overhead dwarfs the
    # actual copy for a single 240x320x3 frame. Stack in numpy first (no GIL
    # release/thread-pool involved) and do the layout conversion exactly once.
    stacked = np.stack(arrays)  # (num_frames, H, W, C) uint8
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
