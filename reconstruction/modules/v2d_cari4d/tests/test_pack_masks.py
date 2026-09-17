# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import av
import h5py
import imageio.v3 as iio
import numpy as np
import pytest

from v2d.cari4d.lib.pack_masks import masks_pack_cari4d_h5


@pytest.fixture
def fragmented_video_and_masks(tmp_path):
    video_path = tmp_path / "example.0.color.mp4"
    with av.open(str(video_path), "w", options={"movflags": "frag_keyframe+empty_moov"}) as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        for index in range(5):
            output.mux(stream.encode(av.VideoFrame.from_ndarray(np.full((12, 16, 3), index * 40, dtype=np.uint8), format="rgb24")))
        output.mux(stream.encode())
    assert iio.improps(video_path, plugin="pyav").shape == (0, 12, 16, 3)
    human_path, object_path = tmp_path / "human", tmp_path / "object"
    for path in (human_path, object_path):
        path.mkdir()
        for index in range(5):
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[index, 1 if path == human_path else 2] = 255
            iio.imwrite(path / f"{index:06d}.png", mask)
    return video_path, human_path, object_path


def test_pack_masks_without_video_frame_count_metadata(tmp_path, fragmented_video_and_masks):
    video, human, obj = fragmented_video_and_masks
    output_path = tmp_path / "masks.h5"
    masks_pack_cari4d_h5(video, human, obj, output_path)
    with h5py.File(output_path, "r") as output:
        assert output.attrs["frame_count"] == 5
        assert output.attrs["width"] == 16
        assert output.attrs["height"] == 12
        assert set(output["example"]) == {f"{index:06d}-k0.{role}.png" for index in range(5) for role in ("person_mask", "obj_rend_mask")}
        for index in range(5):
            assert np.array_equal(output[f"example/{index:06d}-k0.person_mask.png"][:], iio.imread(human / f"{index:06d}.png"))
            assert np.array_equal(output[f"example/{index:06d}-k0.obj_rend_mask.png"][:], iio.imread(obj / f"{index:06d}.png"))


def test_pack_masks_still_rejects_missing_mask_frames(tmp_path, fragmented_video_and_masks):
    video, human, obj = fragmented_video_and_masks
    (obj / "000004.png").unlink()
    output_path = tmp_path / "masks.h5"
    with pytest.raises(ValueError, match="object mask count 4 differs from video frame count 5"):
        masks_pack_cari4d_h5(video, human, obj, output_path)
    assert not output_path.exists()


def test_pack_masks_cli_uses_decoded_frame_count(tmp_path, fragmented_video_and_masks):
    video, human, obj = fragmented_video_and_masks
    output_path = tmp_path / "cli_masks.h5"
    subprocess.run([sys.executable, "-m", "v2d.cari4d.lib.pack_masks", "--video_path", str(video), "--human_masks_path", str(human), "--object_masks_path", str(obj), "--output_path", str(output_path)], check=True)
    with h5py.File(output_path, "r") as output:
        assert output.attrs["frame_count"] == 5
        assert len(output["example"]) == 10
