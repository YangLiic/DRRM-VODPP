from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drrm.datasets.video_frame_decoder import (
    decode_all_video_frames_pyav,
    decode_video_frames_pyav_seek,
)


class _FakeFrame:
    def __init__(self, pts, converted_pts):
        self.pts = pts
        self._converted_pts = converted_pts

    def to_ndarray(self, format):
        assert format == "rgb24"
        self._converted_pts.append(self.pts)
        return np.full((2, 3, 3), self.pts, dtype=np.uint8)


class _FakeContainer:
    def __init__(self, frames, decoded_pts, seek_calls):
        self.streams = SimpleNamespace(
            video=[
                SimpleNamespace(
                    time_base=Fraction(1, 10),
                    codec_context=SimpleNamespace(thread_type=None, thread_count=0),
                )
            ]
        )
        self._frames = frames
        self._decoded_pts = decoded_pts
        self._seek_calls = seek_calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def seek(self, offset, **kwargs):
        self._seek_calls.append((offset, kwargs))

    def decode(self, stream):
        for frame in self._frames:
            self._decoded_pts.append(frame.pts)
            yield frame


class VideoFrameDecoderTest(unittest.TestCase):
    def test_stops_at_last_query_and_converts_only_selected_frames(self):
        decoded_pts = []
        converted_pts = []
        seek_calls = []
        frames = [_FakeFrame(pts, converted_pts) for pts in range(10)]
        container = _FakeContainer(frames, decoded_pts, seek_calls)
        fake_av = SimpleNamespace(open=lambda *args, **kwargs: container)

        with patch.dict("sys.modules", {"av": fake_av}):
            result = decode_video_frames_pyav_seek(
                "episode.mp4", [0.2, 0.4], tolerance_s=0.01
            )

        self.assertEqual(decoded_pts, [0, 1, 2, 3, 4])
        self.assertEqual(converted_pts, [2, 4])
        self.assertEqual(seek_calls[0][0], 2)
        self.assertIs(seek_calls[0][1]["backward"], True)
        self.assertIs(seek_calls[0][1]["any_frame"], False)
        self.assertEqual(result.shape, (2, 3, 2, 3))
        torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([2, 4]) / 255)

    def test_enforces_timestamp_tolerance(self):
        converted_pts = []
        container = _FakeContainer(
            [_FakeFrame(0, converted_pts), _FakeFrame(1, converted_pts)], [], []
        )
        fake_av = SimpleNamespace(open=lambda *args, **kwargs: container)

        with patch.dict("sys.modules", {"av": fake_av}), self.assertRaisesRegex(
            AssertionError, "tolerance"
        ):
            decode_video_frames_pyav_seek("episode.mp4", [0.05], tolerance_s=0.01)

        self.assertEqual(converted_pts, [])


class DecodeAllVideoFramesTest(unittest.TestCase):
    def test_decodes_every_frame_once_in_order_as_uint8(self):
        converted_pts = []
        frames = [_FakeFrame(pts, converted_pts) for pts in range(5)]
        container = _FakeContainer(frames, decoded_pts=[], seek_calls=[])
        fake_av = SimpleNamespace(open=lambda *args, **kwargs: container)

        with patch.dict("sys.modules", {"av": fake_av}):
            result = decode_all_video_frames_pyav("episode.mp4")

        # Every frame is decoded exactly once, in presentation order, with no seeking.
        self.assertEqual(converted_pts, [0, 1, 2, 3, 4])
        self.assertEqual(result.shape, (5, 3, 2, 3))
        self.assertEqual(result.dtype, torch.uint8)
        torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8))
        self.assertEqual(container.streams.video[0].codec_context.thread_type, "AUTO")
        self.assertEqual(container.streams.video[0].codec_context.thread_count, 2)

    def test_raises_on_frame_count_mismatch(self):
        frames = [_FakeFrame(pts, []) for pts in range(3)]
        container = _FakeContainer(frames, decoded_pts=[], seek_calls=[])
        fake_av = SimpleNamespace(open=lambda *args, **kwargs: container)

        with patch.dict("sys.modules", {"av": fake_av}), self.assertRaisesRegex(
            RuntimeError, "expected"
        ):
            decode_all_video_frames_pyav("episode.mp4", expected_num_frames=5)
