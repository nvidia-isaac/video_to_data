"""Offline Phantom DINO + pinned HaMeR inference for bimanual TACO videos."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from . import (
    GROUNDING_DINO_MODEL,
    GROUNDING_DINO_REVISION,
    MANOTORCH_COMMIT,
    PHANTOM_COMMIT,
    PHANTOM_HAMER_COMMIT,
    VITPOSE_COMMIT,
)
from .assets import (
    GROUNDING_DINO_REQUIRED_SHA256,
    HAMER_REQUIRED_SHA256,
    verify_pinned_files,
)
from .checkpoint import load_state_dict_strict
from .geometry import (
    cam_crop_to_full,
    cumulative_mano_joint_rotations,
    mirror_rotation_x,
    project_points,
    project_points_intrinsics,
    remap_virtual_camera_points,
    rotation_matrix_to_wxyz,
)
from .identity import BimanualIdentityTracker, filter_box_area, nms
from .provenance import sha256_file, sha256_tree


RAW_SCHEMA = "v2d.inpainting.phantom-raw/v1"
TRACKING_SCHEMA = "v2d.inpainting.tracking/v1"
RUN_SCHEMA = "v2d.inpainting.phantom-run/v1"
SIDES = ("left", "right")
SIDE_INDEX = {side: index for index, side in enumerate(SIDES)}
SKELETON = tuple(
    (0, first) if offset == 0 else (first + offset - 1, first + offset)
    for first in (1, 5, 9, 13, 17)
    for offset in range(4)
)
COLORS = {"left": (255, 80, 80), "right": (80, 190, 255)}  # RGB
_MANO_TRANSFORMS_TO_JOINTS = (
    0,
    13, 14, 15, 15,
    1, 2, 3, 3,
    4, 5, 6, 6,
    10, 11, 12, 12,
    7, 8, 9, 9,
)


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _read_video(path: Path) -> tuple[list[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Input video decoded zero frames: {path}")
    height, width = frames[0].shape[:2]
    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError(f"Input video has invalid FPS: {fps}")
    if any(frame.shape != (height, width, 3) for frame in frames):
        raise RuntimeError("Input video changes geometry between frames")
    return frames, fps, width, height


def _load_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not load TACO intrinsics from {path}: {error}") from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RuntimeError(f"TACO intrinsics must be one finite 3x3 matrix, got {matrix.shape}")
    if (
        matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[[0, 1], [1, 0]], 0.0, atol=1e-8)
        or not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise RuntimeError("TACO intrinsics are not a valid pinhole matrix")
    if not (0 <= matrix[0, 2] < width and 0 <= matrix[1, 2] < height):
        raise RuntimeError("TACO principal point lies outside the source frame")
    return matrix


def _move_to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _detect_hands(
    frames: list[np.ndarray],
    model_dir: Path,
    *,
    box_threshold: float,
    text_threshold: float,
    nms_iou: float,
    max_candidates: int,
    prompt: str,
    minimum_box_area_fraction: float,
    maximum_box_area_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    detector.eval()
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for frame in tqdm(frames, desc="Grounding DINO", unit="frame"):
        inputs = processor(images=frame, text=prompt, return_tensors="pt")
        inputs = _move_to_device(inputs, device)
        with torch.inference_mode():
            outputs = detector(**inputs)
        try:
            result = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[frame.shape[:2]],
            )[0]
        except TypeError:
            # transformers 4.42 accepted input_ids positionally.
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[frame.shape[:2]],
            )[0]
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        if len(boxes):
            height, width = frame.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
            positive = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores = boxes[positive], scores[positive]
            boxes, scores = filter_box_area(
                boxes,
                scores,
                width,
                height,
                minimum_fraction=minimum_box_area_fraction,
                maximum_fraction=maximum_box_area_fraction,
            )
        if len(boxes):
            kept = nms(boxes, scores, nms_iou)
            boxes, scores = boxes[kept], scores[kept]
            order = np.lexsort((np.arange(len(scores)), -scores))[:max_candidates]
            boxes, scores = boxes[order], scores[order]
        all_boxes.append(boxes)
        all_scores.append(scores)
    del detector, processor
    torch.cuda.empty_cache()
    return all_boxes, all_scores


def _assign_identities(
    boxes_per_frame: list[np.ndarray], width: int, height: int
) -> tuple[list[dict[str, int]], np.ndarray, np.ndarray]:
    tracker = BimanualIdentityTracker()
    assignments: list[dict[str, int]] = []
    ambiguous = np.zeros(len(boxes_per_frame), dtype=np.bool_)
    reasons: list[str] = []
    for frame_index, boxes in enumerate(boxes_per_frame):
        result = tracker.assign(boxes, width, height)
        assignments.append(result.indices)
        ambiguous[frame_index] = result.ambiguous
        reasons.append(result.reason)
    return assignments, ambiguous, np.asarray(reasons, dtype="U64")


def _load_hamer(weights_dir: Path, mano_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    from hamer.configs import get_config
    from hamer.models import HAMER

    checkpoint_path = weights_dir / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    config_path = weights_dir / "_DATA/hamer_ckpts/model_config.yaml"
    mean_path = weights_dir / "_DATA/data/mano_mean_params.npz"
    for required in (
        checkpoint_path,
        config_path,
        mean_path,
        mano_dir / "MANO_LEFT.pkl",
        mano_dir / "MANO_RIGHT.pkl",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    cfg = get_config(str(config_path), update_cachedir=False)
    cfg.defrost()
    cfg.MANO.DATA_DIR = str(weights_dir / "_DATA/data")
    cfg.MANO.MODEL_PATH = str(mano_dir)
    cfg.MANO.MEAN_PARAMS = str(mean_path)
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.freeze()

    # Load the fingerprinted official Lightning dictionary in safe weights-only
    # mode, then apply only its state_dict to the locally constructed model.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("Official HaMeR checkpoint has no state_dict")
    model = HAMER(cfg, init_renderer=False)
    load_report = load_state_dict_strict(model, checkpoint["state_dict"])
    model = model.to(torch.device("cuda:0")).eval()
    del checkpoint
    return model, cfg, load_report


def _load_anatomical_frames(mano_dir: Path) -> dict[str, np.ndarray]:
    """Load native-side AxisLayerFK rest-axis corrections read-only.

    ``manotorch`` expects a root containing ``models/`` while the Phantom
    runner mounts that models directory directly. A temporary symlink adapts
    the layout without copying or modifying the licensed MANO files.
    """

    from manotorch.axislayer import AxisLayerFK

    with tempfile.TemporaryDirectory(prefix="phantom-mano-") as directory:
        root = Path(directory)
        (root / "models").symlink_to(mano_dir, target_is_directory=True)
        result: dict[str, np.ndarray] = {}
        for side in SIDES:
            layer = AxisLayerFK(side=side, mano_assets_root=str(root)).eval()
            frames = layer.TMPL_R_p_a.detach().cpu().numpy()[0].astype(np.float64)
            orthogonality = np.swapaxes(frames, -1, -2) @ frames
            if not np.allclose(
                orthogonality, np.eye(3), atol=2e-5
            ) or not np.allclose(np.linalg.det(frames), 1.0, atol=2e-5):
                raise RuntimeError(f"AxisLayerFK returned invalid {side} rest frames")
            result[side] = frames
    return result


def _empty_raw(
    frame_count: int, width: int, height: int, fps: float, max_candidates: int
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {
        "schema_version": np.asarray(RAW_SCHEMA),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
        "fps": np.asarray(fps, dtype=np.float64),
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
        "side_names": np.asarray(SIDES),
        "detection_count": np.zeros(frame_count, dtype=np.int16),
        "detection_boxes": np.full((frame_count, max_candidates, 4), np.nan, dtype=np.float32),
        "detection_scores": np.full((frame_count, max_candidates), np.nan, dtype=np.float32),
        "identity_ambiguous": np.zeros(frame_count, dtype=np.bool_),
        "identity_reason": np.full(frame_count, "", dtype="U64"),
    }
    for side in SIDES:
        raw[f"{side}_assigned"] = np.zeros(frame_count, dtype=np.bool_)
        raw[f"{side}_valid"] = np.zeros(frame_count, dtype=np.bool_)
        raw[f"{side}_status"] = np.zeros(frame_count, dtype=np.int8)
        raw[f"{side}_box"] = np.full((frame_count, 4), np.nan, dtype=np.float32)
        raw[f"{side}_score"] = np.full(frame_count, np.nan, dtype=np.float32)
        raw[f"{side}_camera_translation"] = np.full((frame_count, 3), np.nan, dtype=np.float32)
        raw[f"{side}_virtual_camera_translation"] = np.full((frame_count, 3), np.nan, dtype=np.float32)
        raw[f"{side}_joints_3d"] = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
        raw[f"{side}_joints_2d"] = np.full((frame_count, 21, 2), np.nan, dtype=np.float32)
        raw[f"{side}_joints_wxyz"] = np.full((frame_count, 21, 4), np.nan, dtype=np.float32)
        raw[f"{side}_vertices_3d"] = np.full((frame_count, 778, 3), np.nan, dtype=np.float32)
        raw[f"{side}_mano_betas"] = np.full((frame_count, 10), np.nan, dtype=np.float32)
        raw[f"{side}_mano_global_orient_rotmat"] = np.full((frame_count, 3, 3), np.nan, dtype=np.float32)
        raw[f"{side}_mano_hand_pose_rotmat"] = np.full((frame_count, 15, 3, 3), np.nan, dtype=np.float32)
    return raw


def _run_hamer(
    frames: list[np.ndarray],
    boxes_per_frame: list[np.ndarray],
    scores_per_frame: list[np.ndarray],
    assignments: list[dict[str, int]],
    ambiguous: np.ndarray,
    reasons: np.ndarray,
    weights_dir: Path,
    mano_dir: Path,
    intrinsics: np.ndarray,
    *,
    fps: float,
    batch_size: int,
    max_candidates: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    height, width = frames[0].shape[:2]
    raw = _empty_raw(len(frames), width, height, fps, max_candidates)
    raw["coordinate_frame"] = np.asarray("camera")
    raw["intrinsics"] = np.asarray(intrinsics, dtype=np.float64)
    raw["identity_ambiguous"] = ambiguous
    raw["identity_reason"] = reasons
    work: list[tuple[int, str, dict[str, Any]]] = []
    for frame_index, (boxes, scores, assignment) in enumerate(
        zip(boxes_per_frame, scores_per_frame, assignments)
    ):
        count = min(len(boxes), max_candidates)
        raw["detection_count"][frame_index] = count
        raw["detection_boxes"][frame_index, :count] = boxes[:count]
        raw["detection_scores"][frame_index, :count] = scores[:count]
        if ambiguous[frame_index]:
            for side in SIDES:
                raw[f"{side}_status"][frame_index] = 2
        for side, candidate_index in assignment.items():
            raw[f"{side}_assigned"][frame_index] = True
            raw[f"{side}_status"][frame_index] = 5  # assigned, inference pending
            raw[f"{side}_box"][frame_index] = boxes[candidate_index]
            raw[f"{side}_score"][frame_index] = scores[candidate_index]
            work.append((frame_index, side, {}))

    model, cfg, load_report = _load_hamer(weights_dir, mano_dir)
    anatomical_frames = _load_anatomical_frames(mano_dir)
    parents = model.mano.parents.detach().cpu().numpy().astype(np.int64)
    if parents.shape != (16,):
        raise RuntimeError(f"Pinned HaMeR MANO parents must have shape (16,), got {parents.shape}")
    scaled_focal = float(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(width, height))
    raw["virtual_focal_length"] = np.asarray(scaled_focal, dtype=np.float64)
    for start in tqdm(range(0, len(work), batch_size), desc="HaMeR", unit="batch"):
        batch_work = work[start : start + batch_size]
        items: list[dict[str, Any]] = []
        for frame_index, side, _ in batch_work:
            box = raw[f"{side}_box"][frame_index][None, :]
            right = np.asarray([side == "right"], dtype=np.float32)
            dataset = ViTDetDataset(
                cfg, frames[frame_index], box, right, rescale_factor=2.0
            )
            items.append(dataset[0])
        batch = {
            key: torch.stack(
                [torch.as_tensor(item[key]) for item in items], dim=0
            ).to("cuda:0")
            for key in ("img", "box_center", "box_size", "img_size", "right")
        }
        with torch.inference_mode():
            output = model(batch)
        pred_cam = output["pred_cam"].detach().cpu().numpy().astype(np.float64)
        right_np = batch["right"].detach().cpu().numpy().reshape(-1)
        pred_cam[:, 1] *= 2.0 * right_np - 1.0
        translations = cam_crop_to_full(
            pred_cam,
            batch["box_center"].detach().cpu().numpy(),
            batch["box_size"].detach().cpu().numpy(),
            batch["img_size"].detach().cpu().numpy(),
            scaled_focal,
        )
        joints = output["pred_keypoints_3d"].detach().cpu().numpy().astype(np.float64)
        vertices = output["pred_vertices"].detach().cpu().numpy().astype(np.float64)
        betas = output["pred_mano_params"]["betas"].detach().cpu().numpy()
        global_orient = (
            output["pred_mano_params"]["global_orient"].detach().cpu().numpy()
        )
        hand_pose = output["pred_mano_params"]["hand_pose"].detach().cpu().numpy()
        for local_index, (frame_index, side, _) in enumerate(batch_work):
            local_joints = joints[local_index].copy()
            local_vertices = vertices[local_index].copy()
            wrist_rotation = global_orient[local_index, 0].copy()
            local_hand_pose = hand_pose[local_index].copy()
            if side == "left":
                local_joints[:, 0] *= -1.0
                local_vertices[:, 0] *= -1.0
                wrist_rotation = mirror_rotation_x(wrist_rotation)
                local_hand_pose = mirror_rotation_x(local_hand_pose)
            translation_camera = remap_virtual_camera_points(
                translations[local_index][None, :],
                scaled_focal,
                width,
                height,
                intrinsics,
            )[0]
            virtual_anchor_uv = project_points(
                translations[local_index][None, :], scaled_focal, width, height
            )
            calibrated_anchor_uv = project_points_intrinsics(
                translation_camera[None, :], intrinsics
            )
            if not np.allclose(calibrated_anchor_uv, virtual_anchor_uv, atol=1e-5):
                raise RuntimeError("TACO focal remap did not preserve the HaMeR camera anchor")
            # Preserve metric MANO articulation. Only the weak-perspective
            # camera anchor is focal-remapped; pointwise remapping would squash
            # the hand and invalidate its rigid kinematics.
            joints_camera = local_joints + translation_camera[None, :]
            vertices_camera = local_vertices + translation_camera[None, :]
            joints_2d = project_points_intrinsics(joints_camera, intrinsics)
            joint_rotations = cumulative_mano_joint_rotations(
                global_orient[local_index, 0],
                hand_pose[local_index],
                parents,
                _MANO_TRANSFORMS_TO_JOINTS,
                mirror_left=side == "left",
                anatomical_frames=anatomical_frames[side],
            )
            joint_quaternions = np.stack(
                [rotation_matrix_to_wxyz(rotation) for rotation in joint_rotations]
            )
            near_edge = (
                (joints_2d[:, 0] < 5)
                | (joints_2d[:, 1] < 5)
                | (joints_2d[:, 0] > width - 5)
                | (joints_2d[:, 1] > height - 5)
            )
            finite = (
                np.isfinite(joints_camera).all()
                and np.isfinite(joints_2d).all()
                and np.isfinite(vertices_camera).all()
                and np.isfinite(wrist_rotation).all()
                and np.isfinite(joint_quaternions).all()
            )
            valid = bool(finite and near_edge.mean() <= 0.1)
            raw[f"{side}_camera_translation"][frame_index] = translation_camera
            raw[f"{side}_virtual_camera_translation"][frame_index] = translations[local_index]
            raw[f"{side}_joints_3d"][frame_index] = joints_camera
            raw[f"{side}_joints_2d"][frame_index] = joints_2d
            raw[f"{side}_joints_wxyz"][frame_index] = joint_quaternions
            raw[f"{side}_vertices_3d"][frame_index] = vertices_camera
            raw[f"{side}_mano_betas"][frame_index] = betas[local_index]
            raw[f"{side}_mano_global_orient_rotmat"][frame_index] = wrist_rotation
            raw[f"{side}_mano_hand_pose_rotmat"][frame_index] = local_hand_pose
            raw[f"{side}_valid"][frame_index] = valid
            raw[f"{side}_status"][frame_index] = 1 if valid else 3
    del model
    torch.cuda.empty_cache()
    return raw, {"scaled_focal_length": scaled_focal, "checkpoint_load": load_report}


def _tracking_from_raw(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    frame_count = len(raw["frame_indices"])
    tracking: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("phantom"),
        "coordinate_frame": np.asarray("camera"),
        "frame_indices": raw["frame_indices"].copy(),
    }
    for side in SIDES:
        valid = raw[f"{side}_valid"].copy()
        quaternions = np.full((frame_count, 4), np.nan, dtype=np.float32)
        for frame_index in np.flatnonzero(valid):
            quaternions[frame_index] = raw[f"{side}_joints_wxyz"][frame_index, 0]
        tracking[f"{side}_valid"] = valid
        tracking[f"{side}_wrist_position"] = raw[f"{side}_joints_3d"][:, 0].copy()
        tracking[f"{side}_wrist_wxyz"] = quaternions
        tracking[f"{side}_joints_3d"] = raw[f"{side}_joints_3d"].copy()
        tracking[f"{side}_joints_2d"] = raw[f"{side}_joints_2d"].copy()
        tracking[f"{side}_joints_wxyz"] = raw[f"{side}_joints_wxyz"].copy()
    return tracking


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _draw_overlay(
    frames: list[np.ndarray], raw: dict[str, np.ndarray], output: Path, fps: float
) -> None:
    temporary = output.with_name(output.stem + ".partial.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        "-framerate",
        f"{fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index, source in enumerate(tqdm(frames, desc="Overlay", unit="frame")):
            frame = source.copy()
            for side in SIDES:
                color = COLORS[side]
                box = raw[f"{side}_box"][frame_index]
                if np.isfinite(box).all():
                    x0, y0, x1, y1 = np.rint(box).astype(int)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
                    score = raw[f"{side}_score"][frame_index]
                    label = f"Phantom {side} {score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (max(0, x0), max(30, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                if raw[f"{side}_valid"][frame_index]:
                    points = np.rint(raw[f"{side}_joints_2d"][frame_index]).astype(int)
                    for first, second in SKELETON:
                        cv2.line(frame, tuple(points[first]), tuple(points[second]), color, 4, cv2.LINE_AA)
                    for point in points:
                        cv2.circle(frame, tuple(point), 4, (255, 255, 255), -1, cv2.LINE_AA)
                elif raw[f"{side}_assigned"][frame_index]:
                    cv2.putText(
                        frame,
                        f"{side}: rejected",
                        (20, 70 + 35 * SIDE_INDEX[side]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
            if raw["identity_ambiguous"][frame_index]:
                cv2.putText(
                    frame,
                    f"AMBIGUOUS: {raw['identity_reason'][frame_index]}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg overlay encoding failed with status {return_code}")
    os.replace(temporary, output)


def _validate_tracking(tracking: dict[str, np.ndarray], frame_count: int) -> None:
    if not np.array_equal(tracking["frame_indices"], np.arange(frame_count)):
        raise RuntimeError("tracking frame indices are not contiguous")
    for side in SIDES:
        valid = tracking[f"{side}_valid"]
        if valid.dtype != np.bool_ or valid.shape != (frame_count,):
            raise RuntimeError(f"{side}_valid violates the tracking contract")
        for key, shape in (
            ("wrist_position", (frame_count, 3)),
            ("wrist_wxyz", (frame_count, 4)),
            ("joints_3d", (frame_count, 21, 3)),
            ("joints_2d", (frame_count, 21, 2)),
            ("joints_wxyz", (frame_count, 21, 4)),
        ):
            value = tracking[f"{side}_{key}"]
            if value.shape != shape or (valid.any() and not np.isfinite(value[valid]).all()):
                raise RuntimeError(f"{side}_{key} violates the tracking contract")
        if valid.any():
            norms = np.linalg.norm(tracking[f"{side}_wrist_wxyz"][valid], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise RuntimeError(f"{side} wrist quaternions are not normalized")
            joint_norms = np.linalg.norm(
                tracking[f"{side}_joints_wxyz"][valid], axis=2
            )
            if not np.allclose(joint_norms, 1.0, atol=1e-3):
                raise RuntimeError(f"{side} joint quaternions are not normalized")


def _validate_npz_roundtrip(
    path: Path, expected: dict[str, np.ndarray], *, tracking: bool = False
) -> None:
    """Verify that a staged archive exactly round-trips before publication."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise RuntimeError(f"{path.name} keys changed during serialization")
            loaded = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not reopen staged archive {path}: {error}") from error
    for name, expected_value in expected.items():
        actual = loaded[name]
        expected_array = np.asarray(expected_value)
        if actual.shape != expected_array.shape or actual.dtype != expected_array.dtype:
            raise RuntimeError(f"{path.name}:{name} changed shape or dtype")
        if expected_array.dtype.kind in {"f", "c"}:
            same_values = np.array_equal(actual, expected_array, equal_nan=True)
        else:
            same_values = np.array_equal(actual, expected_array)
        if not same_values:
            raise RuntimeError(f"{path.name}:{name} changed values during serialization")
    if tracking:
        _validate_tracking(loaded, len(expected["frame_indices"]))


def _validate_overlay(
    path: Path, *, frame_count: int, width: int, height: int, fps: float
) -> None:
    """Decode every staged overlay frame and verify exact video alignment."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen staged overlay: {path}")
    decoded = 0
    try:
        measured_fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Staged overlay geometry differs from the input video")
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise RuntimeError(
            f"Staged overlay decoded {decoded} frames; expected {frame_count}"
        )
    if not np.isfinite(measured_fps) or not np.isclose(
        measured_fps, fps, rtol=1e-4, atol=1e-3
    ):
        raise RuntimeError(f"Staged overlay FPS {measured_fps} differs from input FPS {fps}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Offline Phantom inference requires exactly one visible GPU; "
            f"container sees {torch.cuda.device_count()}"
        )
    _configure_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": args.output_dir / "phantom_raw_predictions.npz",
        "tracking": args.output_dir / "tracking.npz",
        "overlay": args.output_dir / "hand_overlay.mp4",
        "metadata": args.output_dir / "run_metadata.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phantom outputs: {existing}")
    # Metadata is the bundle's commit marker. Invalidate an older complete run
    # before an overwrite can publish any replacement artifacts.
    if args.overwrite:
        outputs["metadata"].unlink(missing_ok=True)

    frames, fps, width, height = _read_video(args.video)
    intrinsics = _load_intrinsics(args.intrinsics, width, height)
    verify_pinned_files(
        args.grounding_dino_dir,
        GROUNDING_DINO_REQUIRED_SHA256,
        asset_name="Grounding DINO",
    )
    boxes, scores = _detect_hands(
        frames,
        args.grounding_dino_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou=args.nms_iou,
        max_candidates=args.max_candidates,
        prompt=args.prompt,
        minimum_box_area_fraction=args.minimum_box_area_fraction,
        maximum_box_area_fraction=args.maximum_box_area_fraction,
    )
    assignments, ambiguous, reasons = _assign_identities(boxes, width, height)
    verify_pinned_files(
        args.hamer_dir,
        HAMER_REQUIRED_SHA256,
        asset_name="HaMeR",
    )
    raw, hamer_report = _run_hamer(
        frames,
        boxes,
        scores,
        assignments,
        ambiguous,
        reasons,
        args.hamer_dir,
        args.mano_dir,
        intrinsics,
        fps=fps,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
    )
    tracking = _tracking_from_raw(raw)
    _validate_tracking(tracking, len(frames))

    # Stage and validate the entire three-file artifact bundle. Publication is
    # a sequence of atomic renames while the metadata commit marker is absent.
    with tempfile.TemporaryDirectory(prefix=".phantom-stage-", dir=args.output_dir) as directory:
        stage = Path(directory)
        staged = {
            "raw": stage / outputs["raw"].name,
            "tracking": stage / outputs["tracking"].name,
            "overlay": stage / outputs["overlay"].name,
        }
        _atomic_savez(staged["raw"], raw)
        _atomic_savez(staged["tracking"], tracking)
        _draw_overlay(frames, raw, staged["overlay"], fps)
        _validate_npz_roundtrip(staged["raw"], raw)
        _validate_npz_roundtrip(staged["tracking"], tracking, tracking=True)
        _validate_overlay(
            staged["overlay"],
            frame_count=len(frames),
            width=width,
            height=height,
            fps=fps,
        )
        for name in ("raw", "tracking", "overlay"):
            os.replace(staged[name], outputs[name])

    valid_fraction = {
        side: float(raw[f"{side}_valid"].mean()) for side in SIDES
    }
    ambiguous_fraction = float(ambiguous.mean())
    quality_failures = [
        f"{side}_valid_fraction={valid_fraction[side]:.6f} < {args.min_valid_fraction:.6f}"
        for side in SIDES
        if valid_fraction[side] < args.min_valid_fraction
    ]
    if ambiguous_fraction > args.max_ambiguous_fraction:
        quality_failures.append(
            f"ambiguous_fraction={ambiguous_fraction:.6f} > {args.max_ambiguous_fraction:.6f}"
        )

    dino_tree_hash, dino_files = sha256_tree(args.grounding_dino_dir)
    hamer_tree_hash, hamer_files = sha256_tree(args.hamer_dir)
    status_counts = {
        side: {
            str(key): int(value)
            for key, value in sorted(Counter(raw[f"{side}_status"].tolist()).items())
        }
        for side in SIDES
    }
    metadata: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "state": "failed_quality" if quality_failures else "complete",
        "sequence_id": args.sequence_id,
        "tracker": "phantom",
        "coordinate_frame": "camera",
        "geometry": {
            "frame_count": len(frames),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "camera_calibration": {
            "coordinate_convention": "opencv_camera_x_right_y_down_z_forward",
            "intrinsics": intrinsics.tolist(),
            "source_path": args.intrinsics_source_path,
            "translation_remap": {
                "policy": "remap_hamer_virtual_camera_anchor_then_add_metric_mano_local_geometry",
                "virtual_focal_length": hamer_report["scaled_focal_length"],
                "depth_scale": float(
                    intrinsics[0, 0] / hamer_report["scaled_focal_length"]
                ),
                "preserves_virtual_camera_anchor_projection": True,
                "pointwise_geometry_scaling": False,
            },
        },
        "joint_rotations": {
            "representation": "normalized_wxyz_quaternion",
            "semantics": "global_anatomy_aligned_mano_joint_rotations_in_camera_axes",
            "axis_convention": "native_side_manotorch_AxisLayerFK.TMPL_R_p_a",
            "mano_transform_to_21_joint_mapping": list(_MANO_TRANSFORMS_TO_JOINTS),
            "left_hand_reflection": "diag(-1,1,1) @ R @ diag(-1,1,1)",
        },
        "implementation": {
            "phantom_commit": PHANTOM_COMMIT,
            "phantom_hamer_commit": PHANTOM_HAMER_COMMIT,
            "vitpose_submodule_commit_not_executed": VITPOSE_COMMIT,
            "manotorch_commit": MANOTORCH_COMMIT,
            "container_image_id": args.container_image_id,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "handedness": {
            "policy": "taco_image_left_is_anatomical_left_then_temporal_assignment",
            "ambiguous_frames_are_invalid": True,
            "uses_ground_truth": False,
            "ambiguous_frames": int(ambiguous.sum()),
            "ambiguous_fraction": ambiguous_fraction,
            "reason_counts": dict(sorted(Counter(reasons.tolist()).items())),
        },
        "parameters": {
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "nms_iou": args.nms_iou,
            "max_candidates": args.max_candidates,
            "minimum_box_area_fraction": args.minimum_box_area_fraction,
            "maximum_box_area_fraction": args.maximum_box_area_fraction,
            "hamer_bbox_rescale_factor": 2.0,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "min_valid_fraction": args.min_valid_fraction,
            "max_ambiguous_fraction": args.max_ambiguous_fraction,
        },
        "quality": {
            "valid_fraction": valid_fraction,
            "status_code_meaning": {
                "0": "not_detected",
                "1": "valid",
                "2": "identity_ambiguous",
                "3": "hamer_edge_or_finite_check_rejected",
                "5": "inference_pending_unexpected",
            },
            "status_counts": status_counts,
            "failures": quality_failures,
        },
        "hamer": hamer_report,
        "inputs": {
            "video": {
                "basename": args.video.name,
                "bytes": args.video.stat().st_size,
                "sha256": sha256_file(args.video),
            },
            "intrinsics": {
                "basename": args.intrinsics.name,
                "source_path": args.intrinsics_source_path,
                "bytes": args.intrinsics.stat().st_size,
                "sha256": sha256_file(args.intrinsics),
                "matrix": intrinsics.tolist(),
            },
            "grounding_dino": {
                "model_id": GROUNDING_DINO_MODEL,
                "revision": GROUNDING_DINO_REVISION,
                "tree_sha256": dino_tree_hash,
                "files": dino_files,
            },
            "hamer": {"tree_sha256": hamer_tree_hash, "files": hamer_files},
            "mano": {
                side: {
                    "bytes": (args.mano_dir / f"MANO_{side.upper()}.pkl").stat().st_size,
                    "sha256": sha256_file(args.mano_dir / f"MANO_{side.upper()}.pkl"),
                }
                for side in SIDES
            },
        },
        "outputs": {
            name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
            if name != "metadata"
        },
        "network": "disabled_during_inference",
        "gpu": "single_container_visible_device_cuda:0",
        "retargeting": {
            "state": "not_run",
            "reason": "Tracker output must pass through the shared Video2Data Sharpa retargeter; no tracker-specific robot semantics are fabricated.",
        },
    }
    _atomic_json(outputs["metadata"], metadata)
    if quality_failures:
        raise RuntimeError("Phantom output failed declared quality gates: " + "; ".join(quality_failures))
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--intrinsics-source-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-dino-dir", type=Path, required=True)
    parser.add_argument("--hamer-dir", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--prompt", default="a hand.")
    parser.add_argument("--box-threshold", type=float, default=0.2)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--minimum-box-area-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-box-area-fraction", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--max-ambiguous-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "box_threshold",
        "text_threshold",
        "nms_iou",
        "min_valid_fraction",
        "max_ambiguous_fraction",
        "minimum_box_area_fraction",
        "maximum_box_area_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.batch_size <= 0 or args.max_candidates < 2:
        parser.error("--batch-size must be positive and --max-candidates must be >= 2")
    if args.minimum_box_area_fraction >= args.maximum_box_area_fraction:
        parser.error("minimum box area fraction must be smaller than maximum")
    return args


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata["quality"], indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Phantom inference failed: {error}", file=sys.stderr)
        raise
