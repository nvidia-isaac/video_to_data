"""CPU-only contract tests for the formal MECKA arm-mask stage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from inpainting.mecka_panda import arm_mask, grounding_dino_runner, sam2_runner
from inpainting.mecka_panda.arm_mask import (
    ArmMaskConfig,
    ArmMaskError,
    _correction_frames,
    _Episode,
    _materialize_window,
    _project_tracks,
    _publish_outputs,
    _tracking_metadata,
    preflight,
)
from inpainting.mecka_panda.contracts import TRACKING_SCHEMA
from inpainting.mecka_panda.video_io import probe_video


def _tracking_arrays(
    frame_count: int = 3, *, coordinate_frame: str = "camera"
) -> dict[str, np.ndarray]:
    points = np.zeros((frame_count, 21, 3), dtype=np.float64)
    points[:, :, 0] = np.linspace(0.05, 0.20, 21)
    points[:, :, 1] = np.linspace(-0.10, 0.10, 21)
    points[:, :, 2] = 1.0
    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("mecka"),
        "coordinate_frame": np.asarray(coordinate_frame),
        "frame_indices": np.arange(frame_count, dtype=np.int64),
    }
    for hand in ("left", "right"):
        result[f"{hand}_valid"] = np.ones(frame_count, dtype=np.bool_)
        result[f"{hand}_wrist_position"] = points[:, 0].copy()
        quaternion = np.zeros((frame_count, 4), dtype=np.float64)
        quaternion[:, 0] = 1.0
        result[f"{hand}_wrist_wxyz"] = quaternion
        result[f"{hand}_joints_3d"] = points.copy()
    return result


def _write_video(path: Path, values: list[int], *, size: tuple[int, int]) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()
    for value in values:
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()


def _box_assignment_tracks() -> dict[str, np.ndarray]:
    offsets = np.linspace(-5.0, 5.0, 21)
    left = np.stack([25.0 + offsets, 50.0 + offsets * 0.2], axis=1)
    right = np.stack([75.0 + offsets, 50.0 + offsets * 0.2], axis=1)
    return {"left": left[None, ...], "right": right[None, ...]}


def _arm_detection(
    x0: float, y0: float, x1: float, y1: float, confidence: float = 0.8
) -> dict[str, object]:
    return {
        "label": "human arm",
        "confidence": confidence,
        "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }


def test_box_assignment_rejects_detection_covering_both_hands() -> None:
    tracks = _box_assignment_tracks()
    broad = _arm_detection(0.0, 0.0, 100.0, 100.0, confidence=0.99)
    left = _arm_detection(0.0, 20.0, 48.0, 100.0)
    right = _arm_detection(52.0, 20.0, 100.0, 100.0)
    present = {
        "left": np.ones(1, dtype=np.bool_),
        "right": np.ones(1, dtype=np.bool_),
    }

    assigned = arm_mask.assign_boxes_to_hands(
        [broad, left, right],
        tracks,
        0,
        100,
        100,
        present,
    )

    assert assigned == {"left": left["box"], "right": right["box"]}
    assert (
        arm_mask.assign_boxes_to_hands(
            [broad],
            tracks,
            0,
            100,
            100,
            present,
        )
        == {}
    )


def test_box_assignment_ignores_other_hand_when_it_is_absent() -> None:
    tracks = _box_assignment_tracks()
    broad = _arm_detection(0.0, 0.0, 100.0, 100.0, confidence=0.99)
    present = {
        "left": np.ones(1, dtype=np.bool_),
        "right": np.zeros(1, dtype=np.bool_),
    }

    assigned = arm_mask.assign_boxes_to_hands(
        [broad],
        tracks,
        0,
        100,
        100,
        present,
    )

    assert assigned == {"left": broad["box"]}


@pytest.mark.parametrize(
    ("hand", "wrist_x", "expected_direction"),
    [("left", 40.0, -1.0), ("right", 60.0, 1.0)],
)
def test_arm_seed_points_extend_from_wrist_toward_image_entry(
    hand: str, wrist_x: float, expected_direction: float
) -> None:
    points = np.full((21, 2), [wrist_x, 30.0], dtype=np.float64)

    seeds = arm_mask._arm_seed_points(points, hand, width=100, height=80)

    assert seeds.shape == (3, 2)
    assert np.all(seeds[:, 1] > points[arm_mask.WRIST, 1])
    assert np.all(
        expected_direction
        * (seeds[:, 0] - points[arm_mask.WRIST, 0])
        > 0
    )
    assert np.all((seeds[:, 0] >= 0) & (seeds[:, 0] < 100))
    assert np.all((seeds[:, 1] >= 0) & (seeds[:, 1] < 80))


def test_estimated_arm_root_follows_wrist_to_palm_axis() -> None:
    points = np.full((21, 2), [60.0, 30.0], dtype=np.float64)
    points[list(arm_mask.PALM[1:])] = [55.0, 20.0]

    root = arm_mask._estimated_arm_root(points, "right", width=100, height=80)
    seeds = arm_mask._arm_seed_points(points, "right", width=100, height=80)

    np.testing.assert_allclose(root, [84.5, 79.0])
    np.testing.assert_allclose(
        seeds[-1],
        points[arm_mask.WRIST] + 0.96 * (root - points[arm_mask.WRIST]),
    )


def test_projection_uses_distortion_and_rejects_world() -> None:
    arrays = _tracking_arrays()
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 110.0, 40.0], [0.0, 0.0, 1.0]])
    distortion = np.asarray([0.2, -0.04, 0.01, 0.002])
    uv, present, _ = _project_tracks(arrays, intrinsic, distortion, ArmMaskConfig())
    expected, _ = cv2.projectPoints(
        arrays["left_joints_3d"][0],
        np.zeros(3),
        np.zeros(3),
        intrinsic,
        distortion,
    )
    np.testing.assert_allclose(uv["left"][0], expected.reshape(21, 2))
    assert present["left"].all()

    world = _tracking_arrays(coordinate_frame="world")
    with pytest.raises(ValueError, match="camera-frame"):
        _project_tracks(world, intrinsic, distortion, ArmMaskConfig())


def test_tracking_metadata_binds_nonzero_window_and_distortion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tracking.json"
    path.write_text(
        json.dumps(
            {
                "state": "complete",
                "frame_window": {"start": 17, "count": 3},
                "video_geometry": {"width": 64, "height": 48},
                "camera_intrinsics_full_resolution": [
                    100.0,
                    100.0,
                    32.0,
                    24.0,
                    0.1,
                    -0.02,
                    0.003,
                    0.004,
                ],
            }
        ),
        encoding="utf-8",
    )
    _, distortion = _tracking_metadata(
        path,
        source_start_frame=17,
        frame_count=3,
        width=64,
        height=48,
    )
    np.testing.assert_allclose(distortion, [0.1, -0.02, 0.003, 0.004])
    with pytest.raises(ValueError, match="source_start_frame"):
        _tracking_metadata(
            path,
            source_start_frame=0,
            frame_count=3,
            width=64,
            height=48,
        )


def test_materialize_window_starts_at_exact_nonzero_frame(tmp_path: Path) -> None:
    source = tmp_path / "sentinels.avi"
    _write_video(source, [0, 30, 60, 90, 120, 150], size=(64, 48))
    output = _materialize_window(
        source,
        tmp_path / "window.mp4",
        source_start_frame=2,
        frame_count=3,
        working_width=64,
    )
    geometry = probe_video(output)
    assert (
        geometry["frame_count"],
        geometry["width"],
        geometry["height"],
    ) == (3, 64, 48)
    capture = cv2.VideoCapture(str(output))
    means: list[float] = []
    for _ in range(3):
        ok, frame = capture.read()
        assert ok
        means.append(float(frame.mean()))
    capture.release()
    np.testing.assert_allclose(means, [60, 90, 120], atol=6)


def test_publish_is_bool_full_resolution_and_exact_length(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.avi"
    _write_video(source, [20, 50, 80, 110], size=(64, 48))
    cleaned = tmp_path / "cleaned"
    for object_id in (1, 2):
        (cleaned / str(object_id)).mkdir(parents=True)
        for index in range(2):
            mask = np.zeros((24, 32), dtype=np.uint8)
            if object_id == 1:
                mask[4:12, 5 + index : 10 + index] = 255
            assert cv2.imwrite(str(cleaned / str(object_id) / f"{index:06d}.png"), mask)
    episode = _Episode(
        sequence_id="sentinel",
        episode_index=7,
        source_video=source,
        clip=tmp_path / "unused.mp4",
        output_dir=tmp_path,
        width=32,
        height=24,
        fps=10.0,
        source_start_frame=1,
        frame_count=2,
        uv={hand: np.zeros((2, 21, 2), dtype=np.float64) for hand in ("left", "right")},
        present={hand: np.ones(2, dtype=np.bool_) for hand in ("left", "right")},
        frame_scores=np.ones(2),
    )
    mask_path = tmp_path / "arm_mask.npy"
    preview_path = tmp_path / "preview.mp4"
    _publish_outputs(
        episode,
        cleaned,
        mask_path,
        preview_path,
        ArmMaskConfig(mask_dilate_iterations=0),
        source_width=64,
        source_height=48,
        source_fps=10.0,
    )
    mask = np.load(mask_path, allow_pickle=False)
    assert mask.dtype == np.bool_
    assert mask.shape == (2, 48, 64)
    assert mask.any()
    preview = probe_video(preview_path)
    assert (
        preview["frame_count"],
        preview["width"],
        preview["height"],
    ) == (2, 64, 48)


def test_correction_uses_interval_midpoint_when_all_scores_are_invalid(
    tmp_path: Path,
) -> None:
    episode = _Episode(
        sequence_id="correction",
        episode_index=0,
        source_video=tmp_path / "source.mp4",
        clip=tmp_path / "clip.mp4",
        output_dir=tmp_path,
        width=32,
        height=24,
        fps=10.0,
        source_start_frame=0,
        frame_count=5,
        uv={hand: np.zeros((5, 21, 2), dtype=np.float64) for hand in ("left", "right")},
        present={hand: np.ones(5, dtype=np.bool_) for hand in ("left", "right")},
        frame_scores=np.full(5, -np.inf),
    )
    quality = {
        "summary": {hand: {"failure_ranges": [[0, 4]]} for hand in ("left", "right")}
    }
    selected = _correction_frames(
        quality,
        episode,
        ArmMaskConfig(),
        excluded={"left": set(), "right": set()},
    )
    assert selected == {2: ["left", "right"]}


def test_preflight_checks_runners_weights_and_container_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_ids = {
        "v2d_grounding_dino": "sha256:" + "1" * 64,
        "v2d_sam2": "sha256:" + "2" * 64,
    }
    inspected: list[str] = []

    def inspect(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        image = command[3]
        inspected.append(image)
        return subprocess.CompletedProcess(command, 0, image_ids[image] + "\n", "")

    monkeypatch.setattr(arm_mask.subprocess, "run", inspect)
    required = (
        tmp_path / ".venv/bin/python",
        tmp_path / "modules/v2d_grounding_dino/docker/run_image_to_object_bboxes.py",
        tmp_path / "modules/v2d_docker/container.py",
        tmp_path / "data/weights/grounding_dino/groundingdino_swint_ogc.pth",
        tmp_path / "data/weights/sam2/sam2.1_hiera_large.pt",
        tmp_path / "data/weights/sam2/sam2.1_hiera_l.yaml",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"present")
    required[0].chmod(0o755)
    assert preflight(tmp_path) == tmp_path.resolve()
    assert inspected == ["v2d_grounding_dino", "v2d_sam2"]
    required[-1].unlink()
    with pytest.raises(FileNotFoundError, match="sam2.1_hiera_l.yaml"):
        preflight(tmp_path)


def test_container_image_resolver_rejects_missing_or_mutable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "No such image")

    monkeypatch.setattr(arm_mask.subprocess, "run", missing)
    with pytest.raises(ArmMaskError, match="v2d_grounding_dino"):
        arm_mask.resolve_container_images()

    def mutable(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "v2d:latest\n", "")

    monkeypatch.setattr(arm_mask.subprocess, "run", mutable)
    with pytest.raises(ArmMaskError, match="invalid immutable ID"):
        arm_mask.resolve_container_images()


def test_arm_subcommands_receive_the_resolved_immutable_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounding_id = "sha256:" + "3" * 64
    sam2_id = "sha256:" + "4" * 64
    episode = _Episode(
        sequence_id="pin",
        episode_index=1,
        source_video=tmp_path / "source.mp4",
        clip=tmp_path / "clip.mp4",
        output_dir=tmp_path / "work",
        width=32,
        height=24,
        fps=10.0,
        source_start_frame=0,
        frame_count=1,
        uv={hand: np.zeros((1, 21, 2)) for hand in ("left", "right")},
        present={hand: np.ones(1, dtype=np.bool_) for hand in ("left", "right")},
        frame_scores=np.ones(1),
    )
    commands: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if str(arm_mask.GROUNDING_DINO_RUNNER) in command:
            output = Path(command[command.index("--output_path") + 1])
            output.write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        arm_mask,
        "_decode_frame",
        lambda *_: np.zeros((24, 32, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(arm_mask.subprocess, "run", run)
    reconstruction = tmp_path / "reconstruction"
    arm_mask._run_detector(
        episode,
        0,
        reconstruction,
        ArmMaskConfig(),
        grounding_id,
    )
    arm_mask._run_sam2(
        episode,
        tmp_path / "prompts.json",
        tmp_path / "masks",
        reconstruction,
        sam2_id,
    )

    assert commands[0][1] == str(arm_mask.GROUNDING_DINO_RUNNER)
    assert commands[0][commands[0].index("--image_id") + 1] == grounding_id
    assert commands[0][commands[0].index("--cache_dir") + 1] == str(
        episode.output_dir / "huggingface_cache"
    )
    assert commands[1][1] == str(arm_mask.SAM2_RUNNER)
    assert commands[1][commands[1].index("--image_id") + 1] == sam2_id


def test_formal_grounding_runner_executes_only_the_supplied_image_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    image_id = "sha256:" + "5" * 64
    monkeypatch.setattr(
        grounding_dino_runner,
        "_runtime",
        lambda: (lambda **kwargs: calls.append(kwargs), "/modules"),
    )
    grounding_dino_runner.run_image_to_object_bboxes(
        image_path="/input.png",
        output_path="/output.json",
        prompt="arm",
        model_dir="/weights",
        image_id=image_id,
        cache_dir=str(tmp_path / "huggingface"),
    )
    assert calls[0]["image"] == image_id
    assert calls[0]["modules_dir"] == "/modules"
    assert calls[0]["env"] == {
        "HOME": "/tmp",
        "HF_HOME": "/huggingface-cache",
        "TRANSFORMERS_CACHE": "/huggingface-cache/transformers",
    }
    assert calls[0]["extra_volumes"] == [
        f"{(tmp_path / 'huggingface').resolve()}:/huggingface-cache:rw"
    ]
    assert (tmp_path / "huggingface").is_dir()
    with pytest.raises(ValueError, match="immutable"):
        grounding_dino_runner.run_image_to_object_bboxes(
            image_path="/input.png",
            output_path="/output.json",
            prompt="arm",
            model_dir="/weights",
            image_id="v2d_grounding_dino:latest",
            cache_dir=str(tmp_path / "invalid"),
        )


def test_formal_sam2_runner_uses_image_id_only_as_docker_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    image_id = "sha256:" + "6" * 64
    video = tmp_path / "input.mp4"
    prompts = tmp_path / "prompts.json"
    weights = tmp_path / "weights"
    output = tmp_path / "masks"
    video.write_bytes(b"video")
    prompts.write_text(
        '{"prompts": [{"object_id": 1, "frame_index": 0, "box": {}}]}',
        encoding="utf-8",
    )
    weights.mkdir()
    monkeypatch.setattr(sam2_runner, "_source_frame_count", lambda _: 2)

    def run_in_container(**kwargs: object) -> None:
        calls.append(kwargs)
        outputs = kwargs["outputs"]
        assert isinstance(outputs, dict)
        staged = Path(str(outputs["masks_dir"]))
        staged.mkdir()
        object_dir = staged / "1"
        object_dir.mkdir()
        for frame_index in range(2):
            (object_dir / f"{frame_index:06d}.png").write_bytes(
                b"\x89PNG\r\n\x1a\nmask"
            )

    monkeypatch.setattr(sam2_runner, "_runtime", lambda: (run_in_container, "/modules"))
    sam2_runner.run_video_to_masks(
        video_path=str(video),
        prompts_path=str(prompts),
        masks_dir=str(output),
        weights_dir=str(weights),
        image_id=image_id,
    )
    assert calls[0]["image"] == image_id
    assert "extra_args" not in calls[0]
    assert calls[0]["network_disabled"] is True
    assert calls[0]["strict_io_isolation"] is True
    assert calls[0]["gpu_device"] == 0
    assert (output / sam2_runner.RUN_GENERATION_FILENAME).is_file()
    manifest = json.loads(
        (output / sam2_runner.RUN_GENERATION_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["expected"] == {"object_ids": [1], "frame_count": 2}
