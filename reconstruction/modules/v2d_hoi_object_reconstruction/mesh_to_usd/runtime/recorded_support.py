"""Infer and validate a support pose from recorded object poses."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from geometry import matrix_to_quaternion_wxyz, vectors_to_rotation_matrix


RECORDED_SUPPORT_SCHEMA_VERSION = 1
DEFAULT_STABLE_WINDOW_FRAMES = 30
DEFAULT_INITIAL_SEARCH_FRAMES = 90
EXACT_MESH_SUPPORT_CALIBRATION = "exact-mesh-recorded-poses"
FOUNDATION_POSE_SUPPORT_CALIBRATION = "foundation-pose-target-tracking"
RECORDED_POSE_CAMERA_MODE = "not-applicable-recorded-poses"
FOUNDATION_POSE_CAMERA_MODE = "multi-view-foundation-pose"


def _recorded_pose_camera_provenance() -> dict[str, str]:
    """Describe why camera attribution does not apply to recorded poses."""

    return {
        "mode": RECORDED_POSE_CAMERA_MODE,
        "reason": "support pose read directly from sequence poses.npy",
    }


@dataclass(frozen=True)
class RecordedSupportPose:
    schema_version: int
    pose_source: str
    fallback_used: bool
    sequence_id: str
    mesh_file_sha256: str
    poses_file_sha256: str
    ground_plane_file_sha256: str
    frame_start: int
    frame_end_exclusive: int
    selected_frame_count: int
    ground_normal_world: tuple[float, float, float]
    local_up: tuple[float, float, float]
    initial_rotation_wxyz: tuple[float, float, float, float]
    median_up_deviation_degrees: float
    max_up_deviation_degrees: float
    median_rotation_deviation_degrees: float
    max_rotation_deviation_degrees: float
    max_translation_deviation_m: float
    thresholds: dict[str, float | int]
    frame_selection: dict[str, str | int] = field(
        default_factory=lambda: {"policy": "explicit"}
    )
    recording_mesh_file_sha256: str | None = None
    mesh_frame_transfer: dict[str, object] = field(
        default_factory=lambda: {
            "policy": "exact-mesh",
            "camera_provenance": _recorded_pose_camera_provenance(),
        }
    )

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_mesh_metadata(mesh_path: Path) -> tuple[Path, str]:
    """Validate the alignment sidecar for one canonical aligned mesh."""

    if mesh_path.name != "output_aligned.glb":
        raise ValueError(
            "recorded-pose transfer requires target and recording meshes named "
            "output_aligned.glb"
        )
    symmetry_path = mesh_path.with_name("output_symmetry.json")
    if not symmetry_path.is_file():
        raise ValueError(
            "recorded-pose transfer requires alignment metadata: "
            f"{symmetry_path}"
        )
    try:
        data = json.loads(symmetry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid alignment metadata: {symmetry_path}") from error
    alignment = data.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"alignment metadata lacks 'alignment': {symmetry_path}")
    centroid = np.asarray(alignment.get("centroid"), dtype=np.float64)
    rotation = np.asarray(alignment.get("rotation"), dtype=np.float64)
    if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
        raise ValueError(f"invalid alignment centroid: {symmetry_path}")
    if rotation.size != 16:
        raise ValueError(f"alignment rotation must contain 16 values: {symmetry_path}")
    rotation = rotation.reshape((4, 4))
    if not np.all(np.isfinite(rotation)) or not np.allclose(
        rotation[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0
    ):
        raise ValueError(f"invalid homogeneous alignment rotation: {symmetry_path}")
    basis = rotation[:3, :3]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4, rtol=0.0):
        raise ValueError(f"alignment rotation is not orthonormal: {symmetry_path}")
    if not math.isclose(float(np.linalg.det(basis)), 1.0, abs_tol=1e-4):
        raise ValueError(f"alignment rotation is not proper: {symmetry_path}")
    return symmetry_path, file_sha256(symmetry_path)


def retarget_recorded_support_pose(
    pose: RecordedSupportPose,
    *,
    recording_mesh_path: str | Path,
    target_mesh_path: str | Path,
) -> RecordedSupportPose:
    """Validate exact-mesh support provenance.

    Cross-mesh identity transfer is intentionally rejected. Different
    reconstruction methods can emit semantically different aligned frames even
    when both files satisfy the ``output_aligned.glb`` naming contract. Use
    target-mesh FoundationPose tracking and
    :func:`foundation_pose_calibrated_support_pose` for cross-mesh support.
    """

    recording_mesh = Path(recording_mesh_path).expanduser().resolve()
    target_mesh = Path(target_mesh_path).expanduser().resolve()
    if not recording_mesh.is_file():
        raise ValueError(f"missing recording mesh: {recording_mesh}")
    if not target_mesh.is_file():
        raise ValueError(f"missing target mesh: {target_mesh}")

    recording_hash = file_sha256(recording_mesh)
    expected_recording_hash = (
        pose.recording_mesh_file_sha256 or pose.mesh_file_sha256
    )
    if recording_hash != expected_recording_hash:
        raise ValueError(
            "recording mesh hash does not match recorded-support provenance"
        )
    target_hash = file_sha256(target_mesh)
    if target_hash == recording_hash:
        return replace(
            pose,
            mesh_file_sha256=target_hash,
            recording_mesh_file_sha256=recording_hash,
            mesh_frame_transfer={
                "policy": "exact-mesh",
                "recording_mesh_file_sha256": recording_hash,
                "target_mesh_file_sha256": target_hash,
                "camera_provenance": _recorded_pose_camera_provenance(),
            },
        )

    _aligned_mesh_metadata(recording_mesh)
    _aligned_mesh_metadata(target_mesh)
    raise ValueError(
        "cross-mesh recorded support requires target-mesh FoundationPose "
        "calibration; identity aligned-frame transfer is not allowed"
    )


def foundation_pose_calibrated_support_pose(
    target_pose: RecordedSupportPose,
    *,
    calibration_report_path: str | Path,
) -> RecordedSupportPose:
    """Attach validated target-mesh FoundationPose calibration provenance."""

    report_path = Path(calibration_report_path).expanduser().resolve()
    if not report_path.is_file():
        raise ValueError(
            f"missing FoundationPose support calibration report: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid FoundationPose support calibration report: {report_path}"
        ) from error
    if report.get("status") != "completed":
        raise ValueError("FoundationPose support calibration did not complete")
    expected = {
        "sequence_id": target_pose.sequence_id,
        "target_mesh_file_sha256": target_pose.mesh_file_sha256,
        "poses_file_sha256": target_pose.poses_file_sha256,
    }
    report_values = {
        "sequence_id": report.get("sequence_id"),
        "target_mesh_file_sha256": report.get("target_mesh_file_sha256"),
        "poses_file_sha256": report.get("poses_file_sha256"),
    }
    mismatched = [
        name
        for name, expected_value in expected.items()
        if report_values[name] != expected_value
    ]
    if mismatched:
        raise ValueError(
            "FoundationPose calibration provenance mismatch: "
            + ", ".join(mismatched)
        )
    tracking_frame_start = report.get("frame_start")
    tracking_frame_end = report.get("frame_end_exclusive")
    if (
        type(tracking_frame_start) is not int
        or type(tracking_frame_end) is not int
        or tracking_frame_start > target_pose.frame_start
        or tracking_frame_end < target_pose.frame_end_exclusive
    ):
        raise ValueError(
            "FoundationPose calibration does not cover the selected support window"
        )
    target_symmetry_hash = report.get("target_symmetry_file_sha256")
    if not isinstance(target_symmetry_hash, str) or len(target_symmetry_hash) != 64:
        raise ValueError(
            "FoundationPose calibration lacks target symmetry provenance"
        )
    camera_provenance = report.get("camera_provenance")
    if not isinstance(camera_provenance, dict):
        raise ValueError(
            "FoundationPose calibration lacks camera provenance"
        )
    camera_names = camera_provenance.get("camera_names")
    registration_camera_names = camera_provenance.get(
        "registration_camera_names"
    )
    highest_visibility_camera = camera_provenance.get(
        "highest_visibility_registration_camera"
    )
    if (
        camera_provenance.get("mode") != FOUNDATION_POSE_CAMERA_MODE
        or not isinstance(camera_names, list)
        or not camera_names
        or not isinstance(registration_camera_names, list)
        or not registration_camera_names
        or any(name not in camera_names for name in registration_camera_names)
        or highest_visibility_camera not in registration_camera_names
    ):
        raise ValueError(
            "FoundationPose calibration contains invalid camera provenance"
        )

    return replace(
        target_pose,
        recording_mesh_file_sha256=target_pose.mesh_file_sha256,
        mesh_frame_transfer={
            "policy": "foundation-pose-target-tracking",
            "local_up_transform": "direct-target-pose-inference",
            "recording_mesh_file_sha256": target_pose.mesh_file_sha256,
            "target_mesh_file_sha256": target_pose.mesh_file_sha256,
            "target_poses_file_sha256": target_pose.poses_file_sha256,
            "target_symmetry_file_sha256": target_symmetry_hash,
            "foundation_pose_report_file_sha256": file_sha256(report_path),
            "camera_provenance": camera_provenance,
        },
    )


def _unit_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain 3 finite values")
    length = float(np.linalg.norm(vector))
    if length <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} cannot be the zero vector")
    return vector / length


def _rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _validate_pose_rotations(rotations: np.ndarray) -> None:
    identity = np.eye(3)
    orthonormal = np.all(
        np.isclose(
            np.einsum("nji,njk->nik", rotations, rotations),
            identity,
            atol=1e-4,
            rtol=0.0,
        ),
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)
    valid = orthonormal & np.isclose(determinants, 1.0, atol=1e-4, rtol=0.0)
    if not np.all(valid):
        bad = np.flatnonzero(~valid)
        raise ValueError(
            "poses contain invalid rotation matrices at selected offsets "
            + ", ".join(str(int(index)) for index in bad[:8])
        )


def poses_to_recorded_support(
    poses: np.ndarray,
    ground_plane: np.ndarray,
    *,
    frame_start: int,
    frame_end_exclusive: int,
    sequence_id: str,
    mesh_file_sha256: str,
    poses_file_sha256: str,
    ground_plane_file_sha256: str,
    max_up_deviation_degrees: float = 2.0,
    max_rotation_deviation_degrees: float = 3.0,
    max_translation_deviation_m: float = 0.02,
    frame_selection: dict[str, str | int] | None = None,
) -> RecordedSupportPose:
    """Infer one strict support pose from an explicitly selected frame window."""

    transforms = np.asarray(poses, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    if not 0 <= frame_start < frame_end_exclusive <= len(transforms):
        raise ValueError("frame range must be non-empty and inside poses.npy")
    if min(
        max_up_deviation_degrees,
        max_rotation_deviation_degrees,
        max_translation_deviation_m,
    ) <= 0:
        raise ValueError("recorded-support stability thresholds must be positive")

    selected_transforms = transforms[frame_start:frame_end_exclusive]
    if not np.all(np.isfinite(selected_transforms)):
        raise ValueError("selected poses must be finite")
    if not np.allclose(
        selected_transforms[:, 3, :],
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("selected poses must be homogeneous rigid transforms")

    selected_count = len(selected_transforms)
    rotations = selected_transforms[:, :3, :3]
    _validate_pose_rotations(rotations)
    plane = np.asarray(ground_plane, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("ground plane must contain 4 finite coefficients")
    ground_normal = _unit_vector(plane[:3], "ground plane normal")

    local_ups = np.einsum("nji,j->ni", rotations, ground_normal)
    mean_local_up = local_ups.mean(axis=0)
    local_up = _unit_vector(mean_local_up, "mean recorded local up")
    up_deviations = np.degrees(
        np.arccos(np.clip(local_ups @ local_up, -1.0, 1.0))
    )
    max_up_deviation = float(np.max(up_deviations))
    if max_up_deviation > max_up_deviation_degrees:
        raise ValueError(
            f"recorded local-up spread {max_up_deviation:.3f} deg exceeds "
            f"{max_up_deviation_degrees:.3f} deg"
        )

    pairwise_angles = np.asarray(
        [
            [_rotation_angle_degrees(first, second) for second in rotations]
            for first in rotations
        ]
    )
    medoid_index = int(np.argmin(np.median(pairwise_angles, axis=1)))
    rotation_deviations = pairwise_angles[medoid_index]
    max_rotation_deviation = float(np.max(rotation_deviations))
    if max_rotation_deviation > max_rotation_deviation_degrees:
        raise ValueError(
            f"recorded rotation spread {max_rotation_deviation:.3f} deg exceeds "
            f"{max_rotation_deviation_degrees:.3f} deg"
        )

    translations = selected_transforms[:, :3, 3]
    median_translation = np.median(translations, axis=0)
    translation_deviations = np.linalg.norm(
        translations - median_translation,
        axis=1,
    )
    max_translation_deviation = float(np.max(translation_deviations))
    if max_translation_deviation > max_translation_deviation_m:
        raise ValueError(
            f"recorded translation spread {max_translation_deviation:.4f} m exceeds "
            f"{max_translation_deviation_m:.4f} m"
        )

    initial_rotation = vectors_to_rotation_matrix(
        local_up,
        np.asarray((0.0, 0.0, 1.0)),
    )
    selection = (
        dict(frame_selection)
        if frame_selection is not None
        else {
            "policy": "explicit",
            "search_start": frame_start,
            "search_end_exclusive": frame_end_exclusive,
            "window_frames": selected_count,
        }
    )
    return RecordedSupportPose(
        schema_version=RECORDED_SUPPORT_SCHEMA_VERSION,
        pose_source="recorded",
        fallback_used=False,
        sequence_id=sequence_id,
        mesh_file_sha256=mesh_file_sha256,
        poses_file_sha256=poses_file_sha256,
        ground_plane_file_sha256=ground_plane_file_sha256,
        frame_start=frame_start,
        frame_end_exclusive=frame_end_exclusive,
        selected_frame_count=selected_count,
        ground_normal_world=tuple(float(value) for value in ground_normal),
        local_up=tuple(float(value) for value in local_up),
        initial_rotation_wxyz=matrix_to_quaternion_wxyz(initial_rotation),
        median_up_deviation_degrees=float(np.median(up_deviations)),
        max_up_deviation_degrees=max_up_deviation,
        median_rotation_deviation_degrees=float(np.median(rotation_deviations)),
        max_rotation_deviation_degrees=max_rotation_deviation,
        max_translation_deviation_m=max_translation_deviation,
        thresholds={
            "max_up_deviation_degrees": max_up_deviation_degrees,
            "max_rotation_deviation_degrees": max_rotation_deviation_degrees,
            "max_translation_deviation_m": max_translation_deviation_m,
        },
        frame_selection=selection,
        recording_mesh_file_sha256=mesh_file_sha256,
        mesh_frame_transfer={
            "policy": "exact-mesh",
            "recording_mesh_file_sha256": mesh_file_sha256,
            "target_mesh_file_sha256": mesh_file_sha256,
            "camera_provenance": _recorded_pose_camera_provenance(),
        },
    )


def find_initial_stable_recorded_support(
    poses: np.ndarray,
    ground_plane: np.ndarray,
    *,
    sequence_id: str,
    mesh_file_sha256: str,
    poses_file_sha256: str,
    ground_plane_file_sha256: str,
    stable_window_frames: int = DEFAULT_STABLE_WINDOW_FRAMES,
    initial_search_frames: int = DEFAULT_INITIAL_SEARCH_FRAMES,
    **thresholds,
) -> RecordedSupportPose:
    """Return the first stable window found in a bounded initial prefix."""

    transforms = np.asarray(poses)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    if stable_window_frames <= 0:
        raise ValueError("stable_window_frames must be positive")
    if initial_search_frames <= 0:
        raise ValueError("initial_search_frames must be positive")
    if stable_window_frames > initial_search_frames:
        raise ValueError(
            "stable_window_frames cannot exceed initial_search_frames"
        )

    search_end = min(initial_search_frames, len(transforms))
    if search_end < stable_window_frames:
        raise ValueError(
            f"initial search has {search_end} frames but a stable segment requires "
            f"{stable_window_frames}; no fallback was attempted"
        )

    first_rejection = None
    last_start = search_end - stable_window_frames
    for frame_start in range(last_start + 1):
        frame_end = frame_start + stable_window_frames
        try:
            return poses_to_recorded_support(
                transforms,
                ground_plane,
                frame_start=frame_start,
                frame_end_exclusive=frame_end,
                sequence_id=sequence_id,
                mesh_file_sha256=mesh_file_sha256,
                poses_file_sha256=poses_file_sha256,
                ground_plane_file_sha256=ground_plane_file_sha256,
                frame_selection={
                    "policy": "initial-stable",
                    "search_start": 0,
                    "search_end_exclusive": search_end,
                    "window_frames": stable_window_frames,
                },
                **thresholds,
            )
        except ValueError as error:
            if first_rejection is None:
                first_rejection = str(error)

    detail = f" First rejection: {first_rejection}." if first_rejection else ""
    raise ValueError(
        "no stable segment of still frames was found in initial frames "
        f"[0, {search_end}); checked {last_start + 1} windows of "
        f"{stable_window_frames} frames.{detail} No fallback was attempted."
    )


def recording_to_support_pose(
    sequence_dir: str | Path,
    *,
    tracked_mesh_path: str | Path | None = None,
    tracked_poses_path: str | Path | None = None,
    tracked_frame_start: int = 0,
    frame_start: int | None = None,
    frame_end_exclusive: int | None = None,
    stable_window_frames: int = DEFAULT_STABLE_WINDOW_FRAMES,
    initial_search_frames: int = DEFAULT_INITIAL_SEARCH_FRAMES,
    **thresholds,
) -> RecordedSupportPose:
    """Load a recording and select an initial stable or explicit support window."""

    sequence = Path(sequence_dir).expanduser().resolve()
    if (tracked_mesh_path is None) != (tracked_poses_path is None):
        raise ValueError(
            "tracked_mesh_path and tracked_poses_path must be provided together"
        )
    if type(tracked_frame_start) is not int or tracked_frame_start < 0:
        raise ValueError("tracked_frame_start must be a non-negative integer")
    if tracked_mesh_path is None and tracked_frame_start != 0:
        raise ValueError(
            "tracked_frame_start requires tracked_mesh_path and "
            "tracked_poses_path"
        )
    mesh_path = (
        Path(tracked_mesh_path).expanduser().resolve()
        if tracked_mesh_path is not None
        else sequence / "object_mesh" / "output_aligned.glb"
    )
    poses_path = (
        Path(tracked_poses_path).expanduser().resolve()
        if tracked_poses_path is not None
        else sequence / "poses.npy"
    )
    ground_plane_path = sequence / "ground_plane.json"
    required = (mesh_path, poses_path, ground_plane_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "recorded-support input is incomplete; missing: " + ", ".join(missing)
        )

    ground_data = json.loads(ground_plane_path.read_text(encoding="utf-8"))
    if "plane" not in ground_data:
        raise ValueError(f"ground plane file lacks 'plane': {ground_plane_path}")
    if (frame_start is None) != (frame_end_exclusive is None):
        raise ValueError(
            "frame_start and frame_end_exclusive must be provided together"
        )

    poses = np.load(poses_path, allow_pickle=False)
    ground_plane = np.asarray(ground_data["plane"], dtype=np.float64)
    provenance = {
        "sequence_id": sequence.name,
        "mesh_file_sha256": file_sha256(mesh_path),
        "poses_file_sha256": file_sha256(poses_path),
        "ground_plane_file_sha256": file_sha256(ground_plane_path),
    }
    if frame_start is not None and frame_end_exclusive is not None:
        local_frame_start = frame_start - tracked_frame_start
        local_frame_end = frame_end_exclusive - tracked_frame_start
        support = poses_to_recorded_support(
            poses,
            ground_plane,
            frame_start=local_frame_start,
            frame_end_exclusive=local_frame_end,
            **provenance,
            **thresholds,
        )
    else:
        support = find_initial_stable_recorded_support(
            poses,
            ground_plane,
            stable_window_frames=stable_window_frames,
            initial_search_frames=initial_search_frames,
            **provenance,
            **thresholds,
        )
    if tracked_frame_start == 0:
        return support

    frame_selection = dict(support.frame_selection)
    for key in ("search_start", "search_end_exclusive"):
        value = frame_selection.get(key)
        if type(value) is int:
            frame_selection[key] = value + tracked_frame_start
    return replace(
        support,
        frame_start=support.frame_start + tracked_frame_start,
        frame_end_exclusive=(
            support.frame_end_exclusive + tracked_frame_start
        ),
        frame_selection=frame_selection,
    )


def recorded_support_pose_save(
    pose: RecordedSupportPose,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pose.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output


def recorded_support_pose_load(path: str | Path) -> RecordedSupportPose:
    """Load a recorded-support JSON without tolerating implicit fallbacks."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"missing recorded-support file: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != RECORDED_SUPPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported recorded-support schema: {data.get('schema_version')}"
        )
    if data.get("pose_source") != "recorded" or data.get("fallback_used") is not False:
        raise ValueError(
            "recorded-support JSON must declare recorded input with no fallback"
        )
    try:
        pose = RecordedSupportPose(**data)
    except TypeError as error:
        raise ValueError(f"invalid recorded-support fields: {error}") from error
    _unit_vector(np.asarray(pose.local_up), "recorded local_up")
    quaternion = np.asarray(pose.initial_rotation_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("initial_rotation_wxyz must contain 4 finite values")
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6):
        raise ValueError("initial_rotation_wxyz must be normalized")
    if len(pose.mesh_file_sha256) != 64:
        raise ValueError("mesh_file_sha256 must be a SHA-256 digest")
    if pose.recording_mesh_file_sha256 is None:
        pose = replace(
            pose,
            recording_mesh_file_sha256=pose.mesh_file_sha256,
            mesh_frame_transfer={
                "policy": "exact-mesh",
                "recording_mesh_file_sha256": pose.mesh_file_sha256,
                "target_mesh_file_sha256": pose.mesh_file_sha256,
                "camera_provenance": _recorded_pose_camera_provenance(),
            },
        )
    if len(pose.recording_mesh_file_sha256) != 64:
        raise ValueError(
            "recording_mesh_file_sha256 must be a SHA-256 digest"
        )
    transfer_policy = pose.mesh_frame_transfer.get("policy")
    if transfer_policy not in {
        "exact-mesh",
        "foundation-pose-target-tracking",
    }:
        raise ValueError("unsupported recorded-support mesh-frame transfer")
    if (
        transfer_policy == "exact-mesh"
        and "camera_provenance" not in pose.mesh_frame_transfer
    ):
        pose = replace(
            pose,
            mesh_frame_transfer={
                **pose.mesh_frame_transfer,
                "camera_provenance": _recorded_pose_camera_provenance(),
            },
        )
    if pose.mesh_frame_transfer.get("target_mesh_file_sha256") not in {
        None,
        pose.mesh_file_sha256,
    }:
        raise ValueError("mesh-frame transfer target hash does not match support mesh")
    if pose.mesh_frame_transfer.get("recording_mesh_file_sha256") not in {
        None,
        pose.recording_mesh_file_sha256,
    }:
        raise ValueError(
            "mesh-frame transfer recording hash does not match support provenance"
        )
    if transfer_policy == "exact-mesh":
        if pose.recording_mesh_file_sha256 != pose.mesh_file_sha256:
            raise ValueError("exact-mesh transfer requires identical mesh hashes")
    else:
        required_hashes = (
            "target_poses_file_sha256",
            "target_symmetry_file_sha256",
            "foundation_pose_report_file_sha256",
        )
        if (
            pose.mesh_frame_transfer.get("local_up_transform")
            != "direct-target-pose-inference"
        ):
            raise ValueError(
                "FoundationPose transfer must use direct target-pose inference"
            )
        if any(
            len(pose.mesh_frame_transfer.get(name, "")) != 64
            for name in required_hashes
        ):
            raise ValueError(
                "FoundationPose transfer requires complete calibration hashes"
            )
        if (
            pose.mesh_frame_transfer.get("target_poses_file_sha256")
            != pose.poses_file_sha256
        ):
            raise ValueError(
                "FoundationPose transfer target poses do not match support poses"
            )
    if not isinstance(pose.frame_selection, dict):
        raise ValueError("frame_selection must be an object")
    policy = pose.frame_selection.get("policy")
    if policy not in {"explicit", "initial-stable"}:
        raise ValueError("unsupported recorded-support frame-selection policy")
    if policy == "initial-stable":
        required = (
            "search_start",
            "search_end_exclusive",
            "window_frames",
        )
        if any(type(pose.frame_selection.get(name)) is not int for name in required):
            raise ValueError("initial-stable frame_selection requires integer bounds")
        search_start = pose.frame_selection["search_start"]
        search_end = pose.frame_selection["search_end_exclusive"]
        if (
            search_start < 0
            or search_start > pose.frame_start
            or search_end < pose.frame_end_exclusive
        ):
            raise ValueError(
                "selected frames fall outside the initial search prefix"
            )
        if pose.frame_selection["window_frames"] != pose.selected_frame_count:
            raise ValueError(
                "initial-stable window size does not match selected frames"
            )
    return pose
