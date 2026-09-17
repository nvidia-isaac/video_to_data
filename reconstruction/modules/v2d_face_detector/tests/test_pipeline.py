from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
COMMON_DIR = Path(__file__).resolve().parents[2] / "v2d_common"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(COMMON_DIR))

from face_detection import FaceDetection
import mv_detect_and_blur_faces as pipeline
from mv_detect_and_blur_faces import (
    MissingCameraInputsError,
    TEMPORARY_INPUT_FAILURE_EXIT_CODE,
    _image_writer,
    _resolve_camera_paths,
    process_camera,
)
from omegaconf import OmegaConf
from video import FrameSource


class FakeDetector:
    def __init__(self):
        self.index = 0

    def detect(self, _frame):
        index = self.index
        self.index += 1
        if index == 1:
            return []
        x = 12 + index
        return [
            FaceDetection(
                bbox_xywh=np.array([x, 8, 18, 20]),
                landmarks=np.array(
                    [[x + 5, 14], [x + 13, 14], [x + 9, 18], [x + 6, 23], [x + 12, 23]]
                ),
                score=0.9,
            )
        ]


def _config():
    return {
        "max_gap_frames": 5,
        "edge_fill_frames": 2,
        "track_min_iou": 0.2,
        "track_max_center_distance": 0.75,
        "median_window": 3,
        "ema_alpha": 0.6,
        "ellipse_scale": 1.0,
        "feather_fraction": 0.12,
        "blur_sigma_fraction": 0.18,
        "fps": 30,
        "video_crf": 17,
    }


def test_default_config_uses_image_directory_outputs():
    config_path = LIB_DIR / "mv_detect_and_blur_faces.yaml"
    cfg = OmegaConf.merge(
        OmegaConf.load(config_path),
        {"rgb_dir": "/rgb", "model_dir": "/models", "output_dir": "/output"},
    )

    assert cfg.output_image_path_template.format(cam_name="front").endswith(
        "/images/front"
    )


def test_local_image_writer_outputs_png_directory(tmp_path):
    output_dir = tmp_path / "images" / "front"
    with _image_writer(output_dir) as writer:
        writer.write_frame(np.zeros((8, 10, 3), np.uint8), stem="frame-a")

    assert (output_dir / "frame-a.png").is_file()


def test_rig_resolves_all_camera_image_directories(tmp_path):
    camera_names = (
        "front_stereo_camera_left",
        "front_stereo_camera_right",
        "back_stereo_camera_left",
        "back_stereo_camera_right",
        "left_stereo_camera_left",
        "left_stereo_camera_right",
        "right_stereo_camera_left",
        "right_stereo_camera_right",
    )
    for camera_name in camera_names:
        (tmp_path / camera_name).mkdir()
    cfg = OmegaConf.create(
        {
            "rig_config": "stereo-4",
            "rgb_dir": str(tmp_path),
            "rgb_path_template": "${rgb_dir}/{cam_name}",
        }
    )

    paths = _resolve_camera_paths(cfg)

    assert tuple(paths) == camera_names
    assert paths["front_stereo_camera_left"] == tmp_path / "front_stereo_camera_left"


def test_rig_h5_template_falls_back_to_legacy_image_directories(tmp_path):
    camera_names = (
        "front_stereo_camera_left",
        "front_stereo_camera_right",
        "back_stereo_camera_left",
        "back_stereo_camera_right",
        "left_stereo_camera_left",
        "left_stereo_camera_right",
        "right_stereo_camera_left",
        "right_stereo_camera_right",
    )
    for camera_name in camera_names:
        (tmp_path / camera_name).mkdir()
    cfg = OmegaConf.create(
        {
            "rig_config": "stereo-4",
            "rgb_dir": str(tmp_path),
            "rgb_path_template": "${rgb_dir}/{cam_name}.h5",
        }
    )

    paths = _resolve_camera_paths(cfg)

    assert tuple(paths) == camera_names
    assert all(path.is_dir() for path in paths.values())


def test_missing_rig_inputs_use_temporary_failure_exit_code(
    tmp_path, monkeypatch, capsys,
):
    cfg = OmegaConf.create(
        {
            "rig_config": "stereo-4",
            "rgb_dir": str(tmp_path),
            "rgb_path_template": "${rgb_dir}/{cam_name}.h5",
        }
    )
    with pytest.raises(MissingCameraInputsError):
        _resolve_camera_paths(cfg)

    monkeypatch.setattr(
        pipeline,
        "mv_detect_and_blur_faces",
        lambda _cfg: (_ for _ in ()).throw(
            MissingCameraInputsError("missing one camera")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mv_detect_and_blur_faces",
            "--rgb_dir",
            str(tmp_path),
            "--model_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        pipeline.main()

    assert exc_info.value.code == TEMPORARY_INPUT_FAILURE_EXIT_CODE
    assert "missing one camera" in capsys.readouterr().err


def test_process_camera_preserves_h5_contract_and_writes_video_and_json(
    tmp_path, capsys
):
    source_path = tmp_path / "input.h5"
    image_output = tmp_path / "images" / "camera.h5"
    video_output = tmp_path / "videos" / "camera.mp4"
    detection_output = tmp_path / "detections" / "camera.json"
    frames = np.random.default_rng(1).integers(
        0, 256, size=(3, 48, 64, 3), dtype=np.uint8
    )
    with h5py.File(source_path, "w") as h5_file:
        h5_file.create_dataset("frames", data=frames)
        h5_file.attrs["stems"] = json.dumps(["frame-a", "frame-b", "frame-c"])

    result = process_camera(
        camera_name="camera",
        rgb_path=source_path,
        image_output_path=image_output,
        video_output_path=video_output,
        detection_output_path=detection_output,
        detector=FakeDetector(),
        config=_config(),
    )

    assert result.frame_count == 3
    assert result.interpolated_count == 1
    assert video_output.stat().st_size > 0
    with FrameSource.from_path(image_output) as source:
        assert source.n_frames == 3
        assert source.stems == ["frame-a", "frame-b", "frame-c"]
        assert source.image_size == (64, 48)
        assert source[0].dtype == np.uint8
        assert not np.array_equal(source[0], frames[0])
    with h5py.File(image_output, "r") as h5_file:
        assert h5_file["frames"].compression == "gzip"
        assert h5_file["frames"].compression_opts == 6
    with FrameSource.from_path(video_output) as source:
        assert source.n_frames == 3
        assert source.image_size == (64, 48)
        decoded_frames = list(source.iter_frames())
        assert len(decoded_frames) == 3
        assert decoded_frames[0].shape == (48, 64, 3)
        assert decoded_frames[0].dtype == np.uint8
    payload = json.loads(detection_output.read_text())
    assert payload["schema"] == "v2d.face_detector.detections.v1"
    assert payload["frames"][1]["filtered"][0]["provenance"] == "interpolated"
    assert detection_output.stat().st_mode & 0o777 == 0o644
    progress_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Face detector camera:")
    ]
    assert progress_lines == [
        *[
            f"Face detector camera: {percent}% (detecting)"
            for percent in range(10, 101, 10)
        ],
        *[
            f"Face detector camera: {percent}% (writing)"
            for percent in range(10, 101, 10)
        ],
    ]
