#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a rigid USD from a mesh, with optional strict recorded support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR / "runtime"))

from docker_runtime import IMAGE_NAME
from foundation_pose_support import (
    DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR,
    FOUNDATION_POSE_SUPPORT_POSES_NAME,
    FOUNDATION_POSE_SUPPORT_REPORT_NAME,
    FOUNDATION_POSE_TRACKING_METADATA_NAME,
    run_foundation_pose_support,
)
from recorded_support import (
    DEFAULT_INITIAL_SEARCH_FRAMES,
    DEFAULT_STABLE_WINDOW_FRAMES,
    EXACT_MESH_SUPPORT_CALIBRATION,
    FOUNDATION_POSE_SUPPORT_CALIBRATION,
    RecordedSupportPose,
    file_sha256,
    foundation_pose_calibrated_support_pose,
    recorded_support_pose_save,
    recording_to_support_pose,
    retarget_recorded_support_pose,
)
from run_mesh_to_usd import GENERATION_REPORT_NAME, run_mesh_to_usd
from validation_docker_runtime import VALIDATION_REPORT_NAME, VALIDATOR_IMAGE_NAME
from workflow_io import write_json


RECORDED_SUPPORT_NAME = "recorded_support_pose.json"
WORKFLOW_REPORT_NAME = "mesh_to_usd_workflow_report.json"


def _support_provenance(
    *,
    support_sequence: Path | None,
    support_pose: RecordedSupportPose | None,
    calibration: str | None,
) -> dict[str, object] | None:
    if support_pose is None:
        return None
    if support_sequence is None or calibration is None:
        raise RuntimeError("recorded support is missing sequence provenance")
    camera_provenance = support_pose.mesh_frame_transfer.get(
        "camera_provenance"
    )
    if not isinstance(camera_provenance, dict):
        raise RuntimeError("recorded support is missing camera provenance")
    return {
        "sequence": str(support_sequence),
        "sequence_id": support_pose.sequence_id,
        "pose_calibration": calibration,
        "selected_frame_start": support_pose.frame_start,
        "selected_frame_end_exclusive": support_pose.frame_end_exclusive,
        "camera_provenance": camera_provenance,
    }


def run_mesh_to_usd_workflow(
    output_dir: str,
    *,
    asset_path: str | None = None,
    sequence_dir: str | None = None,
    support_sequence_dir: str | None = None,
    frame_start: int | None = None,
    frame_end_exclusive: int | None = None,
    stable_window_frames: int = DEFAULT_STABLE_WINDOW_FRAMES,
    initial_search_frames: int = DEFAULT_INITIAL_SEARCH_FRAMES,
    max_up_deviation_degrees: float = 2.0,
    max_rotation_deviation_degrees: float = 3.0,
    max_translation_deviation_m: float = 0.02,
    foundation_pose_weights_dir: str | None = None,
    foundation_pose_config_path: str | None = None,
    foundation_pose_debug: int = 0,
    mass_kg: float | None = None,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.4,
    restitution: float = 0.1,
    max_convex_hulls: int = 16,
    hull_vertex_limit: int = 64,
    coacd_resolution: int = 2000,
    max_decomposition_source_faces: int = 20_000,
    simplify_decomposition_source: bool = True,
    image: str = IMAGE_NAME,
    validate: bool = True,
    validator_image: str = VALIDATOR_IMAGE_NAME,
    cache_dir: str | None = None,
    accept_eula: bool = False,
    dev: bool = False,
    gpu_device: str | None = None,
) -> dict:
    """Run one strict asset or recorded-sequence preparation workflow."""

    if (asset_path is None) == (sequence_dir is None):
        raise ValueError("provide exactly one of asset_path or sequence_dir")
    if support_sequence_dir is not None and asset_path is None:
        raise ValueError("support_sequence_dir requires asset_path input")
    if (
        foundation_pose_weights_dir is not None
        or foundation_pose_config_path is not None
        or foundation_pose_debug != 0
    ) and support_sequence_dir is None:
        raise ValueError(
            "FoundationPose support calibration requires support_sequence_dir"
        )
    if (frame_start is None) != (frame_end_exclusive is None):
        raise ValueError(
            "frame_start and frame_end_exclusive must be provided together"
        )
    if (
        asset_path is not None
        and support_sequence_dir is None
        and frame_start is not None
    ):
        raise ValueError(
            "manual frame bounds require sequence_dir or support_sequence_dir input"
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    support_path = output / RECORDED_SUPPORT_NAME
    workflow_report_path = output / WORKFLOW_REPORT_NAME
    foundation_pose_output = output / "foundation_pose_support"
    support_path.unlink(missing_ok=True)
    workflow_report_path.unlink(missing_ok=True)
    for stale_name in (
        FOUNDATION_POSE_SUPPORT_POSES_NAME,
        FOUNDATION_POSE_SUPPORT_REPORT_NAME,
        FOUNDATION_POSE_TRACKING_METADATA_NAME,
    ):
        (foundation_pose_output / stale_name).unlink(missing_ok=True)

    support_pose = None
    support_sequence = None
    foundation_pose_support = None
    if sequence_dir is not None:
        sequence = Path(sequence_dir).expanduser().resolve()
        support_pose = recording_to_support_pose(
            sequence,
            frame_start=frame_start,
            frame_end_exclusive=frame_end_exclusive,
            stable_window_frames=stable_window_frames,
            initial_search_frames=initial_search_frames,
            max_up_deviation_degrees=max_up_deviation_degrees,
            max_rotation_deviation_degrees=max_rotation_deviation_degrees,
            max_translation_deviation_m=max_translation_deviation_m,
        )
        source_asset = sequence / "object_mesh" / "output_aligned.glb"
        input_mode = "recorded-sequence"
        input_path = sequence
        support_sequence = sequence
    elif support_sequence_dir is not None:
        support_sequence = Path(support_sequence_dir).expanduser().resolve()
        source_asset = Path(asset_path).expanduser().resolve()
        recording_mesh = (
            support_sequence / "object_mesh" / "output_aligned.glb"
        )
        exact_recording_mesh = (
            recording_mesh.is_file()
            and file_sha256(source_asset) == file_sha256(recording_mesh)
            and (support_sequence / "poses.npy").is_file()
        )
        if exact_recording_mesh:
            reference_support_pose = recording_to_support_pose(
                support_sequence,
                frame_start=frame_start,
                frame_end_exclusive=frame_end_exclusive,
                stable_window_frames=stable_window_frames,
                initial_search_frames=initial_search_frames,
                max_up_deviation_degrees=max_up_deviation_degrees,
                max_rotation_deviation_degrees=max_rotation_deviation_degrees,
                max_translation_deviation_m=max_translation_deviation_m,
            )
            support_pose = retarget_recorded_support_pose(
                reference_support_pose,
                recording_mesh_path=recording_mesh,
                target_mesh_path=source_asset,
            )
        else:
            tracking_frame_end_exclusive = (
                frame_end_exclusive
                if frame_end_exclusive is not None
                else initial_search_frames
            )
            minimum_tracking_frames = (
                frame_end_exclusive - frame_start
                if frame_start is not None
                and frame_end_exclusive is not None
                else stable_window_frames
            )
            foundation_pose_support = run_foundation_pose_support(
                str(support_sequence),
                str(source_asset),
                foundation_pose_weights_dir,
                str(foundation_pose_output),
                frame_end_exclusive=tracking_frame_end_exclusive,
                allow_shorter_prefix=frame_end_exclusive is None,
                minimum_output_frames=minimum_tracking_frames,
                config_path=foundation_pose_config_path,
                debug=foundation_pose_debug,
                dev=dev,
            )
            target_support_pose = recording_to_support_pose(
                support_sequence,
                tracked_mesh_path=source_asset,
                tracked_poses_path=foundation_pose_support["output_poses"],
                tracked_frame_start=foundation_pose_support["frame_start"],
                frame_start=frame_start,
                frame_end_exclusive=frame_end_exclusive,
                stable_window_frames=stable_window_frames,
                initial_search_frames=initial_search_frames,
                max_up_deviation_degrees=max_up_deviation_degrees,
                max_rotation_deviation_degrees=max_rotation_deviation_degrees,
                max_translation_deviation_m=max_translation_deviation_m,
            )
            support_pose = foundation_pose_calibrated_support_pose(
                target_support_pose,
                calibration_report_path=foundation_pose_support["report_file"],
            )
        input_mode = "asset-with-recorded-support"
        input_path = source_asset
    else:
        source_asset = Path(asset_path).expanduser().resolve()
        input_mode = "asset"
        input_path = source_asset

    generation = run_mesh_to_usd(
        str(source_asset),
        str(output),
        mass_kg=mass_kg,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        max_convex_hulls=max_convex_hulls,
        hull_vertex_limit=hull_vertex_limit,
        coacd_resolution=coacd_resolution,
        max_decomposition_source_faces=max_decomposition_source_faces,
        simplify_decomposition_source=simplify_decomposition_source,
        image=image,
        validate=validate,
        validator_image=validator_image,
        cache_dir=cache_dir,
        accept_eula=accept_eula,
        dev=dev,
        gpu_device=gpu_device,
    )

    if support_pose is not None:
        generated_source_hash = generation.get("input_asset_file_sha256")
        if generated_source_hash != support_pose.mesh_file_sha256:
            raise RuntimeError(
                "generated USD source hash does not match recorded sequence mesh"
            )
        recorded_support_pose_save(support_pose, support_path)

    support_pose_calibration = (
        FOUNDATION_POSE_SUPPORT_CALIBRATION
        if foundation_pose_support is not None
        else EXACT_MESH_SUPPORT_CALIBRATION
        if support_pose is not None
        else None
    )
    result = {
        "status": "generated",
        "input_mode": input_mode,
        "input": str(input_path),
        "source_asset": str(source_asset),
        "support_sequence": (
            str(support_sequence) if support_sequence is not None else None
        ),
        "output_usd": generation["output_usd"],
        "visual_asset": generation["visual_asset"],
        "generation_report": str(output / GENERATION_REPORT_NAME),
        "validation_report": (
            str(output / VALIDATION_REPORT_NAME)
            if "simready_validation" in generation
            else None
        ),
        "recorded_support_pose": (
            str(support_path) if support_pose is not None else None
        ),
        "support_pose_calibration": support_pose_calibration,
        "support_provenance": _support_provenance(
            support_sequence=support_sequence,
            support_pose=support_pose,
            calibration=support_pose_calibration,
        ),
        "foundation_pose_support_report": (
            foundation_pose_support["report_file"]
            if foundation_pose_support is not None
            else None
        ),
        "foundation_pose_support_poses": (
            foundation_pose_support["output_poses"]
            if foundation_pose_support is not None
            else None
        ),
        "foundation_pose_tracking_metadata": (
            foundation_pose_support["tracking_metadata"]
            if foundation_pose_support is not None
            else None
        ),
        "workflow_report": str(workflow_report_path),
    }
    if "simready_validation" in generation:
        result["simready_validation"] = generation["simready_validation"]
    write_json(workflow_report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--asset", dest="asset_path")
    inputs.add_argument("--sequence-dir")
    parser.add_argument(
        "--support-sequence-dir",
        help=(
            "Exported HOI sequence supplying the ground plane and per-camera "
            "inputs for target-mesh FoundationPose support inference"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int, dest="frame_end_exclusive")
    parser.add_argument(
        "--stable-window-frames",
        type=int,
        default=DEFAULT_STABLE_WINDOW_FRAMES,
    )
    parser.add_argument(
        "--initial-search-frames",
        type=int,
        default=DEFAULT_INITIAL_SEARCH_FRAMES,
    )
    parser.add_argument("--max-up-deviation-degrees", type=float, default=2.0)
    parser.add_argument("--max-rotation-deviation-degrees", type=float, default=3.0)
    parser.add_argument("--max-translation-deviation-m", type=float, default=0.02)
    parser.add_argument(
        "--foundation-pose-weights-dir",
        help=(
            "FoundationPose weights used to track a target mesh against the "
            "support sequence when its bytes differ from the recording mesh "
            f"(default: {DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR})"
        ),
    )
    parser.add_argument(
        "--foundation-pose-config-path",
        help="Optional multi-view FoundationPose configuration override",
    )
    parser.add_argument(
        "--foundation-pose-debug",
        type=int,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--static-friction", type=float, default=0.5)
    parser.add_argument("--dynamic-friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.1)
    parser.add_argument("--max-convex-hulls", type=int, default=16)
    parser.add_argument("--hull-vertex-limit", type=int, default=64)
    parser.add_argument("--coacd-resolution", type=int, default=2000)
    parser.add_argument("--max-decomposition-source-faces", type=int, default=20_000)
    parser.add_argument(
        "--simplify-decomposition-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--image", default=IMAGE_NAME)
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validator-image", default=VALIDATOR_IMAGE_NAME)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--gpu-device",
        help="Expose exactly one GPU to Isaac Sim (for example: 0)",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--dev", action="store_true")
    result = run_mesh_to_usd_workflow(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
