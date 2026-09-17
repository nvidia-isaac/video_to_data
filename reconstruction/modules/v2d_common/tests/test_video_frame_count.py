# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import imageio.v3 as iio
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video


@pytest.mark.parametrize("suffix,options", [(".mp4", {"movflags": "frag_keyframe+empty_moov"}), (".mkv", {})])
def test_video_without_frame_count_metadata(tmp_path, suffix, options):
    path = tmp_path / f"video{suffix}"
    with av.open(str(path), "w", options=options) as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        for index in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), index * 40, dtype=np.uint8), format="rgb24")
            output.mux(stream.encode(frame))
        output.mux(stream.encode())

    assert iio.improps(path, plugin="pyav").shape == (0, 12, 16, 3)
    assert video.get_video_lwh(path) == (5, 16, 12)
    with video.FrameSource.from_path(path) as source:
        assert source.n_frames == 5
        assert source.image_size == (16, 12)
        assert source.stems == [f"{index:06d}" for index in range(5)]
        decoded = list(source.iter_frames())
        assert len(decoded) == source.n_frames
        batches = list(source.iter_batches(2))
        assert [(start, len(frames)) for start, frames in batches] == [(0, 2), (2, 2), (4, 1)]
        assert np.array_equal(np.stack(decoded), np.stack([frame for _, frames in batches for frame in frames]))


def test_known_frame_count_does_not_decode(monkeypatch):
    monkeypatch.setattr(video.iio, "improps", lambda *args, **kwargs: SimpleNamespace(shape=(5, 12, 16, 3)))

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Known frame counts must retain the metadata-only fast path")

    monkeypatch.setattr(video.av, "open", unexpected_decode)
    assert video.get_video_lwh(Path("video.mp4")) == (5, 16, 12)


@pytest.mark.parametrize("unknown_count", [0, -1, np.inf, np.nan])
def test_unknown_frame_count_is_decoded(monkeypatch, unknown_count):
    monkeypatch.setattr(video.iio, "improps", lambda *args, **kwargs: SimpleNamespace(shape=(unknown_count, 12, 16, 3)))
    container = SimpleNamespace(decode=lambda **kwargs: iter([object(), object()]))
    monkeypatch.setattr(video.av, "open", lambda *args, **kwargs: nullcontext(container))
    assert video.get_video_lwh(Path("video.mp4")) == (2, 16, 12)


def test_empty_decoded_stream_keeps_zero_count(monkeypatch):
    monkeypatch.setattr(video.iio, "improps", lambda *args, **kwargs: SimpleNamespace(shape=(0, 12, 16, 3)))
    container = SimpleNamespace(decode=lambda **kwargs: iter(()))
    monkeypatch.setattr(video.av, "open", lambda *args, **kwargs: nullcontext(container))
    assert video.get_video_lwh(Path("empty.mp4")) == (0, 16, 12)


def test_frame_count_does_not_hide_decoder_errors(monkeypatch):
    monkeypatch.setattr(video.iio, "improps", lambda *args, **kwargs: SimpleNamespace(shape=(0, 12, 16, 3)))

    def decode(**kwargs):
        yield object()
        raise RuntimeError("decoder rejected a packet")

    monkeypatch.setattr(video.av, "open", lambda *args, **kwargs: nullcontext(SimpleNamespace(decode=decode)))
    with pytest.raises(RuntimeError, match="decoder rejected a packet"):
        video.get_video_lwh(Path("corrupt.mp4"))
